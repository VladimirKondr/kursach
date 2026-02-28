"""
OpenTelemetry Integration for IMPALA RL

This package provides comprehensive observability for the IMPALA RL pipeline:
- Distributed tracing across Actor-Learner-NATS components
- Performance metrics (latency, throughput, resource utilization)
- Structured logging with trace correlation
- SigNoz backend integration

Usage:
    from reinvent.runmodes.impala_rl.telemetry import setup_telemetry, get_tracer, get_meter, get_logger
    
    # Initialize telemetry (call once at startup)
    setup_telemetry(env="dev")
    
    # Get instrumentation objects
    tracer = get_tracer(__name__)
    meter = get_meter(__name__)
    logger = get_logger(__name__)
"""

from typing import Optional
import logging as stdlib_logging

from .setup import (
    setup_telemetry,
    shutdown_telemetry,
    is_telemetry_enabled,
)

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Tracer
    from opentelemetry.metrics import Meter
    
    def get_tracer(name: str) -> Tracer:
        """Get a tracer instance for creating spans."""
        return trace.get_tracer(name)
    
    def get_meter(name: str) -> Meter:
        """Get a meter instance for recording metrics."""
        return metrics.get_meter(name)

except ImportError:
    # Graceful degradation if OpenTelemetry not installed
    stdlib_logging.warning(
        "OpenTelemetry not installed. Telemetry features disabled. "
        "Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )
    
    def get_tracer(name: str):
        """Dummy tracer when OpenTelemetry not available."""
        return None
    
    def get_meter(name: str):
        """Dummy meter when OpenTelemetry not available."""
        return None


def get_logger(name: str) -> stdlib_logging.Logger:
    """Get a structured logger with trace context."""
    from .logs import get_structured_logger
    return get_structured_logger(name)


__all__ = [
    "setup_telemetry",
    "shutdown_telemetry",
    "is_telemetry_enabled",
    "get_tracer",
    "get_meter",
    "get_logger",
]
