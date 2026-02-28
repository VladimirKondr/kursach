"""
OpenTelemetry SDK Setup and Initialization

Handles centralized initialization of TracerProvider, MeterProvider, and LoggerProvider.
Supports graceful degradation and multiple backends (SigNoz, console, noop).
"""

import atexit
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Optional

from .config import TelemetryConfig, load_config_from_yaml, get_default_config

# Global state
_telemetry_initialized = False
_telemetry_enabled = False
_config: Optional[TelemetryConfig] = None

logger = logging.getLogger(__name__)


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled and initialized."""
    return _telemetry_enabled


def _get_resource_attributes(config: TelemetryConfig) -> dict:
    """Collect resource attributes for the service."""
    from platform import system, node, python_version
    import uuid
    
    attributes = {
        "service.name": config.service.name,
        "service.version": config.service.version,
        "deployment.environment": config.service.environment,
        "host.name": node(),
        "host.type": system(),
        "process.pid": os.getpid(),
        "process.executable.name": sys.executable,
        "telemetry.sdk.language": "python",
        "telemetry.sdk.version": python_version(),
    }
    
    # Add run_id for filtering test runs in SigNoz
    # Can be set via environment variable, otherwise generate unique ID
    if run_id := os.getenv("IMPALA_RUN_ID"):
        attributes["run.id"] = run_id
    else:
        # Generate unique run ID based on timestamp + random
        attributes["run.id"] = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Add Kubernetes attributes if running in k8s
    if k8s_pod := os.getenv("K8S_POD_NAME"):
        attributes["k8s.pod.name"] = k8s_pod
    if k8s_namespace := os.getenv("K8S_NAMESPACE"):
        attributes["k8s.namespace"] = k8s_namespace
    if k8s_node := os.getenv("K8S_NODE_NAME"):
        attributes["k8s.node.name"] = k8s_node
    if k8s_deployment := os.getenv("K8S_DEPLOYMENT_NAME"):
        attributes["k8s.deployment.name"] = k8s_deployment
    
    # Add custom resource attributes from config
    attributes.update(config.traces.resource_attributes)
    
    return attributes


def _setup_tracer_provider(config: TelemetryConfig, resource) -> bool:
    """Initialize TracerProvider with OTLP exporter."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_ON,
            ALWAYS_OFF,
            ParentBased,
            TraceIdRatioBased,
        )
        
        # Select sampler based on config
        sampler_map = {
            "always_on": ALWAYS_ON,
            "always_off": ALWAYS_OFF,
            "parent_based": ParentBased(root=TraceIdRatioBased(config.traces.sampling.ratio)),
            "trace_id_ratio": TraceIdRatioBased(config.traces.sampling.ratio),
        }
        sampler = sampler_map.get(config.traces.sampling.type, ALWAYS_ON)
        
        # Create provider
        provider = TracerProvider(resource=resource, sampler=sampler)
        
        # Setup exporter
        if config.traces.exporter.type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                
                exporter = OTLPSpanExporter(
                    endpoint=config.traces.exporter.endpoint,
                    insecure=config.traces.exporter.insecure,
                    headers=tuple(config.traces.exporter.headers.items()) if config.traces.exporter.headers else None,
                    timeout=config.traces.exporter.timeout_seconds,
                )
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                logger.info(f"OTLP trace exporter configured: {config.traces.exporter.endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup OTLP trace exporter: {e}. Falling back to console.")
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
        elif config.traces.exporter.type == "console":
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
        # noop: no processor added
        
        trace.set_tracer_provider(provider)
        logger.info("TracerProvider initialized successfully")
        return True
        
    except ImportError as e:
        logger.error(f"OpenTelemetry trace SDK not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize TracerProvider: {e}")
        return False


def _setup_meter_provider(config: TelemetryConfig, resource) -> bool:
    """Initialize MeterProvider with OTLP exporter."""
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
        from opentelemetry.sdk.metrics.view import View
        
        # Create metric readers
        readers = []
        
        if config.metrics.exporter.type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                
                exporter = OTLPMetricExporter(
                    endpoint=config.metrics.exporter.endpoint,
                    insecure=config.metrics.exporter.insecure,
                    headers=tuple(config.metrics.exporter.headers.items()) if config.metrics.exporter.headers else None,
                    timeout=config.metrics.exporter.timeout_seconds,
                )
                reader = PeriodicExportingMetricReader(
                    exporter,
                    export_interval_millis=config.metrics.interval_seconds * 1000,
                )
                readers.append(reader)
                logger.info(f"OTLP metric exporter configured: {config.metrics.exporter.endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup OTLP metric exporter: {e}. Falling back to console.")
                reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=config.metrics.interval_seconds * 1000,
                )
                readers.append(reader)
        elif config.metrics.exporter.type == "console":
            reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=config.metrics.interval_seconds * 1000,
            )
            readers.append(reader)
        
        # Setup histogram views with custom boundaries
        # Note: Views need to be set up with the provider, not added later
        # For simplicity, we'll use default boundaries and document how to customize
        
        provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(provider)
        logger.info("MeterProvider initialized successfully")
        return True
        
    except ImportError as e:
        logger.error(f"OpenTelemetry metrics SDK not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize MeterProvider: {e}")
        return False


def _setup_logger_provider(config: TelemetryConfig, resource) -> bool:
    """Initialize LoggerProvider with OTLP exporter."""
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
        
        provider = LoggerProvider(resource=resource)
        
        if config.logs.exporter.type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
                
                exporter = OTLPLogExporter(
                    endpoint=config.logs.exporter.endpoint,
                    insecure=config.logs.exporter.insecure,
                    headers=tuple(config.logs.exporter.headers.items()) if config.logs.exporter.headers else None,
                    timeout=config.logs.exporter.timeout_seconds,
                )
                processor = BatchLogRecordProcessor(exporter)
                provider.add_log_record_processor(processor)
                logger.info(f"OTLP log exporter configured: {config.logs.exporter.endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup OTLP log exporter: {e}. Falling back to console.")
                processor = BatchLogRecordProcessor(ConsoleLogExporter())
                provider.add_log_record_processor(processor)
        elif config.logs.exporter.type == "console":
            processor = BatchLogRecordProcessor(ConsoleLogExporter())
            provider.add_log_record_processor(processor)
        
        set_logger_provider(provider)
        
        # Attach OTEL handler to root logger
        handler = LoggingHandler(level=getattr(logging, config.logs.level), logger_provider=provider)
        logging.getLogger().addHandler(handler)
        
        logger.info("LoggerProvider initialized successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"OpenTelemetry logs SDK not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize LoggerProvider: {e}")
        return False


def _test_endpoint_connectivity(endpoint: str, max_retries: int = 3) -> bool:
    """Test connectivity to OTLP endpoint with retries."""
    # Extract host and port from endpoint
    # Format: http://host:port or host:port
    endpoint_clean = endpoint.replace("http://", "").replace("https://", "")
    
    if ":" in endpoint_clean:
        host, port_str = endpoint_clean.split(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            logger.warning(f"Invalid port in endpoint: {endpoint}")
            return False
    else:
        host = endpoint_clean
        port = 4317  # Default OTLP gRPC port
    
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"Successfully connected to {endpoint}")
                return True
            else:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} to {endpoint} failed")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.warning(f"Connection test to {endpoint} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return False


def setup_telemetry(
    config_path: Optional[str] = None,
    env: Optional[str] = None,
    test_connectivity: bool = True,
) -> bool:
    """
    Initialize OpenTelemetry SDK with the specified configuration.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.
        env: Environment name (dev/staging/prod). Overrides config file.
        test_connectivity: Whether to test connectivity to OTLP endpoint before setup.
    
    Returns:
        bool: True if telemetry was successfully initialized, False otherwise.
    """
    global _telemetry_initialized, _telemetry_enabled, _config
    
    if _telemetry_initialized:
        logger.warning("Telemetry already initialized. Skipping.")
        return _telemetry_enabled
    
    # Check if telemetry is disabled via env var
    if os.getenv("IMPALA_TELEMETRY_ENABLED", "true").lower() == "false":
        logger.info("Telemetry disabled via IMPALA_TELEMETRY_ENABLED environment variable")
        _telemetry_initialized = True
        _telemetry_enabled = False
        return False
    
    try:
        # Load configuration
        if config_path and Path(config_path).exists():
            logger.info(f"Loading telemetry config from: {config_path}")
            config = load_config_from_yaml(Path(config_path))
        else:
            env = env or os.getenv("IMPALA_ENV", "dev")
            logger.info(f"Using default telemetry config for environment: {env}")
            config = get_default_config(env)
        
        _config = config
        
        # Test connectivity if enabled
        if test_connectivity and config.traces.exporter.type == "otlp":
            if not _test_endpoint_connectivity(config.traces.exporter.endpoint):
                logger.warning(
                    f"Cannot reach OTLP endpoint {config.traces.exporter.endpoint}. "
                    "Telemetry will use fallback exporters."
                )
                config.traces.exporter.type = "console"
                config.metrics.exporter.type = "console"
                config.logs.exporter.type = "console"
        
        # Setup Resource
        try:
            from opentelemetry.sdk.resources import Resource
            resource_attrs = _get_resource_attributes(config)
            resource = Resource.create(resource_attrs)
            logger.info(f"Resource created: {config.service.name} v{config.service.version}")
        except ImportError:
            logger.error("OpenTelemetry SDK not available")
            _telemetry_initialized = True
            _telemetry_enabled = False
            return False
        
        # Initialize providers
        success = True
        if config.traces.enabled:
            success = success and _setup_tracer_provider(config, resource)
        if config.metrics.enabled:
            success = success and _setup_meter_provider(config, resource)
        if config.logs.enabled:
            success = success and _setup_logger_provider(config, resource)
        
        # Register shutdown hooks
        atexit.register(shutdown_telemetry)
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_telemetry())
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_telemetry())
        
        _telemetry_initialized = True
        _telemetry_enabled = success
        
        if success:
            logger.info("✓ Telemetry initialization complete")
        else:
            logger.warning("⚠ Telemetry partially initialized with fallbacks")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}", exc_info=True)
        _telemetry_initialized = True
        _telemetry_enabled = False
        return False


def shutdown_telemetry():
    """Shutdown telemetry providers and flush remaining data."""
    global _telemetry_initialized, _telemetry_enabled
    
    if not _telemetry_initialized or not _telemetry_enabled:
        return
    
    logger.info("Shutting down telemetry...")
    
    try:
        from opentelemetry import trace, metrics
        
        # Shutdown trace provider
        if hasattr(trace, "get_tracer_provider"):
            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        
        # Shutdown meter provider
        if hasattr(metrics, "get_meter_provider"):
            provider = metrics.get_meter_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        
        # Shutdown logger provider
        try:
            from opentelemetry._logs import get_logger_provider
            provider = get_logger_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except ImportError:
            pass
        
        logger.info("Telemetry shutdown complete")
    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")
    
    _telemetry_enabled = False


def get_config() -> Optional[TelemetryConfig]:
    """Get the current telemetry configuration."""
    return _config
