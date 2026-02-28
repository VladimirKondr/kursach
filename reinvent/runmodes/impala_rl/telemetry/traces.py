"""
Distributed Tracing Utilities

Decorators and context managers for instrumenting code with OpenTelemetry spans.
Thread-safe and async-safe implementations with automatic exception recording.
"""

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from .setup import is_telemetry_enabled

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, handle gracefully if not available
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span, Tracer
    from opentelemetry.trace.propagation import get_current_span
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry trace API not available. Trace decorators will be no-ops.")


def _safe_add_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """Safely add attributes to span, handling non-string values."""
    if not hasattr(span, "set_attribute"):
        return
    
    for key, value in attributes.items():
        try:
            # Convert non-primitive types to strings
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            elif value is None:
                span.set_attribute(key, "null")
            else:
                span.set_attribute(key, str(value))
        except Exception as e:
            logger.debug(f"Failed to set span attribute {key}: {e}")


def _safe_record_exception(span: Any, exception: Exception) -> None:
    """Safely record exception on span."""
    try:
        if hasattr(span, "record_exception"):
            span.record_exception(exception)
        if hasattr(span, "set_status"):
            span.set_status(Status(StatusCode.ERROR, str(exception)))
    except Exception as e:
        logger.debug(f"Failed to record exception on span: {e}")


def _safe_add_event(span: Any, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Safely add event to span."""
    try:
        if hasattr(span, "add_event"):
            span.add_event(name, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to add event to span: {e}")


@contextmanager
def create_span(
    name: str,
    tracer: Optional[Any] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
):
    """
    Context manager for creating a span inline.
    
    Example:
        with create_span("custom_operation", attributes={"key": "value"}) as span:
            # do work
            span.add_event("checkpoint_reached")
            span.set_attribute("result", result_value)
    
    Args:
        name: Name of the span
        tracer: Optional tracer instance. If None, gets tracer from context.
        attributes: Optional attributes to add to the span
        kind: Optional span kind (INTERNAL, CLIENT, SERVER, etc.)
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        # No-op context manager
        class DummySpan:
            def set_attribute(self, key, value): pass
            def add_event(self, name, attributes=None): pass
            def set_status(self, status): pass
            def record_exception(self, exception): pass
        
        yield DummySpan()
        return
    
    try:
        if tracer is None:
            # Get tracer from the calling module
            frame = inspect.currentframe()
            if frame and frame.f_back:
                module_name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                module_name = __name__
            tracer = trace.get_tracer(module_name)
        
        span_kwargs = {}
        if kind is not None:
            span_kwargs["kind"] = kind
        
        with tracer.start_as_current_span(name, **span_kwargs) as span:
            if attributes:
                _safe_add_attributes(span, attributes)
            
            try:
                yield span
            except Exception as e:
                _safe_record_exception(span, e)
                raise
    except Exception as e:
        logger.debug(f"Error in create_span: {e}")
        # Yield dummy span on error
        class DummySpan:
            def set_attribute(self, key, value): pass
            def add_event(self, name, attributes=None): pass
            def set_status(self, status): pass
            def record_exception(self, exception): pass
        
        yield DummySpan()


def trace_operation(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    capture_args: bool = False,
):
    """
    Decorator for tracing synchronous functions/methods.
    
    Example:
        @trace_operation(name="my_operation", attributes={"component": "actor"})
        def my_function(arg1, arg2):
            # function body
            pass
    
    Args:
        name: Span name. If None, uses function name.
        attributes: Static attributes to add to all spans.
        record_exception: Whether to record exceptions on span.
        capture_args: Whether to capture function arguments as span attributes.
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE or not callable(func):
            return func
        
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_telemetry_enabled():
                return func(*args, **kwargs)
            
            try:
                tracer = trace.get_tracer(func.__module__)
                
                with tracer.start_as_current_span(span_name) as span:
                    # Add static attributes
                    if attributes:
                        _safe_add_attributes(span, attributes)
                    
                    # Capture function arguments if requested
                    if capture_args:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        for arg_name, arg_value in bound_args.arguments.items():
                            if arg_name != "self" and arg_name != "cls":
                                _safe_add_attributes(span, {f"arg.{arg_name}": arg_value})
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        if record_exception:
                            _safe_record_exception(span, e)
                        raise
            except Exception as e:
                # If tracing itself fails, log and continue with function execution
                logger.debug(f"Tracing failed for {span_name}: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def trace_async_operation(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    capture_args: bool = False,
):
    """
    Decorator for tracing asynchronous functions/methods.
    
    Example:
        @trace_async_operation(name="async_operation")
        async def my_async_function():
            # async function body
            await something()
    
    Args:
        name: Span name. If None, uses function name.
        attributes: Static attributes to add to all spans.
        record_exception: Whether to record exceptions on span.
        capture_args: Whether to capture function arguments as span attributes.
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE or not callable(func):
            return func
        
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_telemetry_enabled():
                return await func(*args, **kwargs)
            
            try:
                tracer = trace.get_tracer(func.__module__)
                
                with tracer.start_as_current_span(span_name) as span:
                    # Add static attributes
                    if attributes:
                        _safe_add_attributes(span, attributes)
                    
                    # Capture function arguments if requested
                    if capture_args:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        for arg_name, arg_value in bound_args.arguments.items():
                            if arg_name != "self" and arg_name != "cls":
                                _safe_add_attributes(span, {f"arg.{arg_name}": arg_value})
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        if record_exception:
                            _safe_record_exception(span, e)
                        raise
            except Exception as e:
                # If tracing itself fails, log and continue with function execution
                logger.debug(f"Tracing failed for {span_name}: {e}")
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context for propagation.
    
    Returns:
        Dictionary with trace_id and span_id, or empty dict if no active span.
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return {}
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "trace_flags": format(ctx.trace_flags, "02x"),
            }
    except Exception as e:
        logger.debug(f"Failed to get trace context: {e}")
    
    return {}


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    """
    Add attributes to the current active span.
    
    Args:
        attributes: Dictionary of attributes to add
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            _safe_add_attributes(span, attributes)
    except Exception as e:
        logger.debug(f"Failed to add span attributes: {e}")


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current active span.
    
    Args:
        name: Event name
        attributes: Optional event attributes
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            _safe_add_event(span, name, attributes)
    except Exception as e:
        logger.debug(f"Failed to add span event: {e}")


def record_exception_on_current_span(exception: Exception) -> None:
    """
    Record an exception on the current active span.
    
    Args:
        exception: Exception to record
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            _safe_record_exception(span, exception)
    except Exception as e:
        logger.debug(f"Failed to record exception: {e}")
