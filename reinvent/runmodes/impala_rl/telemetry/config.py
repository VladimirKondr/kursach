"""
Telemetry Configuration Models

Pydantic models for validating and managing telemetry configuration.
Supports environment variable overrides and multiple deployment environments.
"""

import os
from typing import Dict, List, Literal
from pathlib import Path

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to dataclasses if Pydantic not available
    from dataclasses import dataclass, field as dataclass_field


if PYDANTIC_AVAILABLE:
    class ExporterConfig(BaseModel):
        """Configuration for OTLP exporter."""
        type: Literal["otlp", "console", "noop"] = "otlp"
        endpoint: str = "http://localhost:4317"
        insecure: bool = True
        headers: Dict[str, str] = Field(default_factory=dict)
        timeout_seconds: int = 10
        
        @field_validator("endpoint")
        @classmethod
        def validate_endpoint(cls, v: str) -> str:
            if not v:
                raise ValueError("Endpoint cannot be empty")
            # Allow env var override
            return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", v)

    
    class SamplingConfig(BaseModel):
        """Trace sampling configuration."""
        type: Literal["always_on", "always_off", "parent_based", "trace_id_ratio"] = "always_on"
        ratio: float = Field(default=1.0, ge=0.0, le=1.0)
        
        @field_validator("ratio")
        @classmethod
        def validate_ratio(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError("Sampling ratio must be between 0.0 and 1.0")
            return v


    class TracesConfig(BaseModel):
        """Trace configuration."""
        enabled: bool = True
        exporter: ExporterConfig = Field(default_factory=ExporterConfig)
        sampling: SamplingConfig = Field(default_factory=SamplingConfig)
        resource_attributes: Dict[str, str] = Field(default_factory=dict)


    class HistogramConfig(BaseModel):
        """Histogram view configuration."""
        boundaries: List[float] = Field(
            default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        @field_validator("boundaries")
        @classmethod
        def validate_boundaries(cls, v: List[float]) -> List[float]:
            if not v:
                raise ValueError("Histogram boundaries cannot be empty")
            if not all(v[i] < v[i+1] for i in range(len(v)-1)):
                raise ValueError("Histogram boundaries must be sorted in ascending order")
            return v


    class MetricsConfig(BaseModel):
        """Metrics configuration."""
        enabled: bool = True
        exporter: ExporterConfig = Field(default_factory=ExporterConfig)
        interval_seconds: int = Field(default=30, ge=5)
        views: HistogramConfig = Field(default_factory=HistogramConfig)


    class LogsConfig(BaseModel):
        """Logs configuration."""
        enabled: bool = True
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
        format: Literal["json", "text"] = "json"
        exporter: ExporterConfig = Field(default_factory=ExporterConfig)
        include_trace_context: bool = True


    class ServiceConfig(BaseModel):
        """Service identification configuration."""
        name: str = "reinvent-impala"
        version: str = "1.0.0"
        environment: Literal["dev", "staging", "prod"] = "dev"
        
        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            if not v:
                raise ValueError("Service name cannot be empty")
            # Allow env var override
            return os.getenv("OTEL_SERVICE_NAME", v)


    class TelemetryConfig(BaseModel):
        """Root telemetry configuration."""
        service: ServiceConfig = Field(default_factory=ServiceConfig)
        traces: TracesConfig = Field(default_factory=TracesConfig)
        metrics: MetricsConfig = Field(default_factory=MetricsConfig)
        logs: LogsConfig = Field(default_factory=LogsConfig)
        
        @model_validator(mode="after")
        def apply_env_overrides(self) -> "TelemetryConfig":
            """Apply environment variable overrides."""
            # OTEL standard env vars
            if env_endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
                self.traces.exporter.endpoint = env_endpoint
                self.metrics.exporter.endpoint = env_endpoint
                self.logs.exporter.endpoint = env_endpoint
            
            if env_sampler := os.getenv("OTEL_TRACES_SAMPLER"):
                valid_samplers = ["always_on", "always_off", "parent_based", "trace_id_ratio"]
                if env_sampler in valid_samplers:
                    self.traces.sampling.type = env_sampler
            
            if env_ratio := os.getenv("OTEL_TRACES_SAMPLER_ARG"):
                try:
                    self.traces.sampling.ratio = float(env_ratio)
                except ValueError:
                    pass
            
            if env_interval := os.getenv("OTEL_METRICS_EXPORT_INTERVAL"):
                try:
                    self.metrics.interval_seconds = int(env_interval)
                except ValueError:
                    pass
            
            # Custom env vars
            if os.getenv("IMPALA_TELEMETRY_ENABLED", "").lower() == "false":
                self.traces.enabled = False
                self.metrics.enabled = False
                self.logs.enabled = False
            
            return self

else:
    # Fallback dataclass-based config if Pydantic not available
    @dataclass
    class ExporterConfig:
        type: str = "otlp"
        endpoint: str = "http://localhost:4317"
        insecure: bool = True
        headers: Dict[str, str] = dataclass_field(default_factory=dict)
        timeout_seconds: int = 10

    @dataclass
    class SamplingConfig:
        type: str = "always_on"
        ratio: float = 1.0

    @dataclass
    class TracesConfig:
        enabled: bool = True
        exporter: ExporterConfig = dataclass_field(default_factory=ExporterConfig)
        sampling: SamplingConfig = dataclass_field(default_factory=SamplingConfig)
        resource_attributes: Dict[str, str] = dataclass_field(default_factory=dict)

    @dataclass
    class HistogramConfig:
        boundaries: List[float] = dataclass_field(
            default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

    @dataclass
    class MetricsConfig:
        enabled: bool = True
        exporter: ExporterConfig = dataclass_field(default_factory=ExporterConfig)
        interval_seconds: int = 30
        views: HistogramConfig = dataclass_field(default_factory=HistogramConfig)

    @dataclass
    class LogsConfig:
        enabled: bool = True
        level: str = "INFO"
        format: str = "json"
        exporter: ExporterConfig = dataclass_field(default_factory=ExporterConfig)
        include_trace_context: bool = True

    @dataclass
    class ServiceConfig:
        name: str = "reinvent-impala"
        version: str = "1.0.0"
        environment: str = "dev"

    @dataclass
    class TelemetryConfig:
        service: ServiceConfig = dataclass_field(default_factory=ServiceConfig)
        traces: TracesConfig = dataclass_field(default_factory=TracesConfig)
        metrics: MetricsConfig = dataclass_field(default_factory=MetricsConfig)
        logs: LogsConfig = dataclass_field(default_factory=LogsConfig)


def load_config_from_yaml(config_path: Path) -> TelemetryConfig:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    if PYDANTIC_AVAILABLE:
        return TelemetryConfig(**data)
    else:
        # Simple nested dict to dataclass conversion
        # This is a simplified version - production code would need more robust handling
        return TelemetryConfig(**data)


def get_default_config(env: str = "dev") -> TelemetryConfig:
    """Get default configuration for the specified environment."""
    config = TelemetryConfig()
    
    if env == "prod":
        # Production defaults
        config.traces.sampling.type = "parent_based"
        config.traces.sampling.ratio = 0.1
        config.logs.level = "INFO"
        config.service.environment = "prod"
    elif env == "staging":
        config.traces.sampling.ratio = 0.5
        config.logs.level = "INFO"
        config.service.environment = "staging"
    else:
        # Dev defaults
        config.traces.sampling.type = "always_on"
        config.logs.level = "DEBUG"
        config.service.environment = "dev"
    
    return config
