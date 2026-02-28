"""
Structured Logging with Trace Correlation

Provides JSON-formatted logging with automatic trace context injection.
Integrates with OpenTelemetry for logs-traces correlation in SigNoz.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

from .setup import is_telemetry_enabled, get_config

# Context variables for storing contextual attributes
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})

# Try to import OpenTelemetry
try:
    from opentelemetry.trace import get_current_span
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "reinvent-impala", include_trace_context: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_trace_context = include_trace_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self.service_name,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add trace context if available
        if self.include_trace_context and OTEL_AVAILABLE and is_telemetry_enabled():
            try:
                span = get_current_span()
                if span and span.is_recording():
                    ctx = span.get_span_context()
                    log_data["trace_id"] = format(ctx.trace_id, "032x")
                    log_data["span_id"] = format(ctx.span_id, "016x")
                    log_data["trace_flags"] = format(ctx.trace_flags, "02x")
            except Exception:
                pass  # Silently ignore trace context errors
        
        # Add contextual attributes
        try:
            context_attrs = _log_context.get()
            if context_attrs:
                log_data["attributes"] = context_attrs
        except Exception:
            pass
        
        # Add extra fields from record
        if hasattr(record, "attributes"):
            log_data["attributes"] = {**log_data.get("attributes", {}), **record.attributes}
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with trace context."""
    
    def __init__(self, include_trace_context: bool = True):
        # Use a format with placeholder for trace context
        fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        super().__init__(fmt)
        self.include_trace_context = include_trace_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with trace context."""
        original_msg = record.msg
        
        # Add trace context to message if available
        if self.include_trace_context and OTEL_AVAILABLE and is_telemetry_enabled():
            try:
                span = get_current_span()
                if span and span.is_recording():
                    ctx = span.get_span_context()
                    trace_id = format(ctx.trace_id, "032x")
                    span_id = format(ctx.span_id, "016x")
                    record.msg = f"[trace_id={trace_id} span_id={span_id}] {original_msg}"
            except Exception:
                pass
        
        result = super().format(record)
        record.msg = original_msg  # Restore original message
        return result


class StructuredLogger:
    """
    Wrapper around standard Python logger with structured logging support.
    
    Provides automatic trace context injection and contextual attributes.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _log_with_attributes(self, level: int, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log message with optional attributes."""
        if attributes:
            # Create a LogRecord with custom attributes
            extra = kwargs.get("extra", {})
            extra["attributes"] = attributes
            kwargs["extra"] = extra
        
        self._logger.log(level, msg, **kwargs)
    
    def debug(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message."""
        self._log_with_attributes(logging.DEBUG, msg, attributes, **kwargs)
    
    def info(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message."""
        self._log_with_attributes(logging.INFO, msg, attributes, **kwargs)
    
    def warning(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message."""
        self._log_with_attributes(logging.WARNING, msg, attributes, **kwargs)
    
    def error(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message."""
        self._log_with_attributes(logging.ERROR, msg, attributes, **kwargs)
    
    def critical(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message."""
        self._log_with_attributes(logging.CRITICAL, msg, attributes, **kwargs)
    
    def exception(self, msg: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self._log_with_attributes(logging.ERROR, msg, attributes, **kwargs)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        StructuredLogger instance
    """
    logger = logging.getLogger(name)
    
    # Configure formatter if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        config = get_config()
        if config and config.logs.format == "json":
            formatter = JSONFormatter(
                service_name=config.service.name,
                include_trace_context=config.logs.include_trace_context,
            )
        else:
            formatter = TextFormatter(include_trace_context=True)
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set log level from config
        if config:
            logger.setLevel(getattr(logging, config.logs.level))
        else:
            logger.setLevel(logging.INFO)
    
    return StructuredLogger(logger)


class log_context:
    """
    Context manager for adding contextual attributes to all logs within a scope.
    
    Example:
        with log_context(actor_id=0, model_version=5):
            logger.info("Processing batch")  # Automatically includes actor_id and model_version
    """
    
    def __init__(self, **attributes):
        self.attributes = attributes
        self.token = None
        self.previous_context = None
    
    def __enter__(self):
        # Get current context and merge with new attributes
        self.previous_context = _log_context.get()
        merged_context = {**self.previous_context, **self.attributes}
        self.token = _log_context.set(merged_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        if self.token:
            _log_context.reset(self.token)


def add_log_attributes(**attributes):
    """
    Add attributes to the current log context.
    
    These attributes will be included in all subsequent log messages
    until the context is cleared or exited.
    
    Args:
        **attributes: Key-value pairs to add to log context
    """
    current = _log_context.get()
    _log_context.set({**current, **attributes})


def clear_log_context():
    """Clear all attributes from the current log context."""
    _log_context.set({})


def get_log_context() -> Dict[str, Any]:
    """Get the current log context attributes."""
    return _log_context.get().copy()


# Convenience function to setup a basic logger quickly
def setup_basic_logger(name: str, level: str = "INFO", use_json: bool = True) -> StructuredLogger:
    """
    Setup a basic structured logger with sensible defaults.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON formatting
    
    Returns:
        StructuredLogger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add new handler with formatter
    handler = logging.StreamHandler(sys.stdout)
    
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return StructuredLogger(logger)
