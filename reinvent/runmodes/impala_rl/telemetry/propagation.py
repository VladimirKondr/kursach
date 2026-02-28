"""
Trace Context Propagation for NATS

Implements W3C TraceContext propagation across NATS messages.
Enables distributed tracing from Actor through NATS to Learner.
"""

import logging
from typing import Any, Dict, Optional

from .setup import is_telemetry_enabled

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace, context
    from opentelemetry.trace import get_current_span, SpanContext, TraceFlags, set_span_in_context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Trace context propagation disabled.")


class NATSTraceContextPropagator:
    """
    Propagates W3C TraceContext through NATS message headers.
    
    Uses the W3C Trace Context format:
    - traceparent: <version>-<trace-id>-<parent-id>-<trace-flags>
    - tracestate: <vendor-specific-data>
    """
    
    def __init__(self):
        if OTEL_AVAILABLE:
            self._propagator = TraceContextTextMapPropagator()
        else:
            self._propagator = None
    
    def inject_context_into_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Inject current trace context into NATS message headers.
        
        Args:
            headers: Existing headers dict. If None, creates new dict.
        
        Returns:
            Headers dict with injected trace context
        """
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not self._propagator:
            return headers or {}
        
        try:
            carrier = headers or {}
            
            # Get current context and inject into carrier
            current_context = context.get_current()
            self._propagator.inject(carrier, current_context)
            
            return carrier
        except Exception as e:
            logger.debug(f"Failed to inject trace context: {e}")
            return headers or {}
    
    def extract_context_from_headers(self, headers: Optional[Dict[str, str]]) -> Any:
        """
        Extract trace context from NATS message headers.
        
        Args:
            headers: NATS message headers
        
        Returns:
            OpenTelemetry Context object with extracted trace context
        """
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not self._propagator:
            return None
        
        if not headers:
            return None
        
        try:
            # Extract context from carrier
            extracted_context = self._propagator.extract(carrier=headers)
            return extracted_context
        except Exception as e:
            logger.debug(f"Failed to extract trace context: {e}")
            return None
    
    def create_span_from_headers(
        self,
        tracer: Any,
        span_name: str,
        headers: Optional[Dict[str, str]],
        kind: Optional[Any] = None,
    ) -> Any:
        """
        Create a new span as a child of the trace context in headers.
        
        This is used on the receiving side (Learner) to continue the distributed trace.
        
        Args:
            tracer: OpenTelemetry Tracer instance
            span_name: Name for the new span
            headers: NATS message headers containing trace context
            kind: Optional span kind
        
        Returns:
            Span context manager
        """
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not tracer:
            from contextlib import nullcontext
            return nullcontext()
        
        try:
            # Extract parent context
            parent_context = self.extract_context_from_headers(headers)
            
            # Create span with extracted context as parent
            span_kwargs = {}
            if kind is not None:
                span_kwargs["kind"] = kind
            
            if parent_context:
                span_kwargs["context"] = parent_context
            
            return tracer.start_as_current_span(span_name, **span_kwargs)
        except Exception as e:
            logger.debug(f"Failed to create span from headers: {e}")
            from contextlib import nullcontext
            return nullcontext()


# Global propagator instance
_global_propagator: Optional[NATSTraceContextPropagator] = None


def get_propagator() -> NATSTraceContextPropagator:
    """Get or create the global trace context propagator."""
    global _global_propagator
    if _global_propagator is None:
        _global_propagator = NATSTraceContextPropagator()
    return _global_propagator


def inject_trace_context(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Inject current trace context into headers dict.
    
    Use this when publishing messages to NATS from Actor.
    
    Example:
        headers = inject_trace_context()
        await nats.publish("subject", data, headers=headers)
    
    Args:
        headers: Existing headers. If None, creates new dict.
    
    Returns:
        Headers with injected trace context
    """
    return get_propagator().inject_context_into_headers(headers)


def extract_trace_context(headers: Optional[Dict[str, str]]) -> Any:
    """
    Extract trace context from headers.
    
    Use this when receiving messages from NATS in Learner.
    
    Args:
        headers: NATS message headers
    
    Returns:
        OpenTelemetry Context or None
    """
    return get_propagator().extract_context_from_headers(headers)


def create_linked_span(
    tracer: Any,
    span_name: str,
    headers: Optional[Dict[str, str]],
    kind: Optional[Any] = None,
) -> Any:
    """
    Create a span linked to the trace context in headers.
    
    Use this in Learner to create a child span that continues the trace from Actor.
    
    Example:
        from reinvent.runmodes.impala_rl.telemetry import get_tracer
        from reinvent.runmodes.impala_rl.telemetry.propagation import create_linked_span
        
        tracer = get_tracer(__name__)
        with create_linked_span(tracer, "learner.process_trajectories", nats_headers):
            # Processing code
            pass
    
    Args:
        tracer: OpenTelemetry Tracer instance
        span_name: Name for the new span
        headers: NATS message headers
        kind: Optional span kind
    
    Returns:
        Span context manager
    """
    return get_propagator().create_span_from_headers(tracer, span_name, headers, kind)


def get_trace_id() -> Optional[str]:
    """
    Get the trace ID of the current span.
    
    Returns:
        Hex string trace ID or None if no active span
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return None
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            return format(ctx.trace_id, "032x")
    except Exception as e:
        logger.debug(f"Failed to get trace ID: {e}")
    
    return None


def get_span_id() -> Optional[str]:
    """
    Get the span ID of the current span.
    
    Returns:
        Hex string span ID or None if no active span
    """
    if not OTEL_AVAILABLE or not is_telemetry_enabled():
        return None
    
    try:
        span = get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            return format(ctx.span_id, "016x")
    except Exception as e:
        logger.debug(f"Failed to get span ID: {e}")
    
    return None


def create_trace_context_dict() -> Dict[str, str]:
    """
    Create a dict with current trace context for logging/debugging.
    
    Returns:
        Dict with trace_id, span_id, and trace_flags
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
        logger.debug(f"Failed to create trace context dict: {e}")
    
    return {}
