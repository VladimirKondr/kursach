"""
Metrics Collection and Recording

Decorators and utilities for recording OpenTelemetry metrics.
Supports Counters, Gauges, and Histograms with automatic attribute management.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

from .setup import is_telemetry_enabled

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, handle gracefully if not available
try:
    from opentelemetry import metrics
    from opentelemetry.metrics import Counter, Histogram, UpDownCounter, ObservableGauge
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry metrics API not available. Metrics will not be collected.")


class MetricsCollector:
    """
    Centralized metrics collector for IMPALA components.
    
    Maintains references to all metrics instruments and provides
    type-safe recording methods.
    """
    
    def __init__(self, meter_name: str = "reinvent.impala"):
        self._meter_name = meter_name
        self._meter = None
        self._instruments: Dict[str, Any] = {}
        self._gauge_values: Dict[tuple, float] = {}  # Track previous gauge values for delta calculation
        
        if OTEL_AVAILABLE:
            self._meter = metrics.get_meter(meter_name)
    
    def _get_or_create_counter(self, name: str, unit: str = "1", description: str = "") -> Any:
        """Get or create a Counter instrument."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not self._meter:
            return None
        
        key = f"counter:{name}"
        if key not in self._instruments:
            try:
                self._instruments[key] = self._meter.create_counter(
                    name=name,
                    unit=unit,
                    description=description,
                )
            except Exception as e:
                logger.debug(f"Failed to create counter {name}: {e}")
                return None
        
        return self._instruments.get(key)
    
    def _get_or_create_histogram(self, name: str, unit: str = "s", description: str = "") -> Any:
        """Get or create a Histogram instrument."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not self._meter:
            return None
        
        key = f"histogram:{name}"
        if key not in self._instruments:
            try:
                self._instruments[key] = self._meter.create_histogram(
                    name=name,
                    unit=unit,
                    description=description,
                )
            except Exception as e:
                logger.debug(f"Failed to create histogram {name}: {e}")
                return None
        
        return self._instruments.get(key)
    
    def _get_or_create_updown_counter(self, name: str, unit: str = "1", description: str = "") -> Any:
        """Get or create an UpDownCounter instrument (for gauges that can go up/down)."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled() or not self._meter:
            return None
        
        key = f"updown:{name}"
        if key not in self._instruments:
            try:
                self._instruments[key] = self._meter.create_up_down_counter(
                    name=name,
                    unit=unit,
                    description=description,
                )
            except Exception as e:
                logger.debug(f"Failed to create updown counter {name}: {e}")
                return None
        
        return self._instruments.get(key)
    
    def record_duration(
        self,
        metric_name: str,
        duration: float,
        attributes: Optional[Dict[str, Any]] = None,
        unit: str = "s",
    ) -> None:
        """Record a duration measurement (gauge - current value)."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled():
            return
        
        try:
            gauge = self._get_or_create_updown_counter(
                metric_name,
                unit=unit,
                description=f"Duration of {metric_name}",
            )
            if gauge:
                # Track by metric name + attributes to calculate delta
                attrs_key = tuple(sorted((attributes or {}).items()))
                key = (metric_name, attrs_key)
                
                prev_value = self._gauge_values.get(key, 0.0)
                delta = duration - prev_value
                
                gauge.add(delta, attributes=attributes or {})
                self._gauge_values[key] = duration
        except Exception as e:
            logger.debug(f"Failed to record duration {metric_name}: {e}")
    
    def record_value(
        self,
        metric_name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, Any]] = None,
        unit: str = "1",
        snapshot: bool = False,
    ) -> None:
        """
        Record a value measurement (gauge - current value).
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            attributes: Optional attributes for the metric
            unit: Unit of measurement
            snapshot: If True, record as absolute value (no delta tracking).
                     Use for values like version numbers where you want the actual value,
                     not cumulative deltas. Default False for backward compatibility.
        """
        if not OTEL_AVAILABLE or not is_telemetry_enabled():
            return
        
        try:
            gauge = self._get_or_create_updown_counter(
                metric_name,
                unit=unit,
                description=f"Value of {metric_name}",
            )
            if gauge:
                # Track by metric name + attributes to calculate delta
                attrs_key = tuple(sorted((attributes or {}).items()))
                key = (metric_name, attrs_key)
                
                prev_value = self._gauge_values.get(key, 0.0)
                delta = value - prev_value
                
                gauge.add(delta, attributes=attributes or {})
                self._gauge_values[key] = value
        except Exception as e:
            logger.debug(f"Failed to record value {metric_name}: {e}")
    
    def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        attributes: Optional[Dict[str, Any]] = None,
        unit: str = "1",
    ) -> None:
        """Increment a counter."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled():
            return
        
        try:
            counter = self._get_or_create_counter(
                metric_name,
                unit=unit,
                description=f"Count of {metric_name}",
            )
            if counter:
                counter.add(value, attributes=attributes or {})
        except Exception as e:
            logger.debug(f"Failed to increment counter {metric_name}: {e}")
    
    def set_gauge(
        self,
        metric_name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, Any]] = None,
        unit: str = "1",
    ) -> None:
        """Set a gauge value (uses UpDownCounter with delta calculation)."""
        if not OTEL_AVAILABLE or not is_telemetry_enabled():
            return
        
        try:
            gauge = self._get_or_create_updown_counter(
                metric_name,
                unit=unit,
                description=f"Gauge for {metric_name}",
            )
            if gauge:
                # Track by metric name + attributes to calculate delta
                attrs_key = tuple(sorted((attributes or {}).items()))
                key = (metric_name, attrs_key)
                
                prev_value = self._gauge_values.get(key, 0.0)
                delta = value - prev_value
                
                gauge.add(delta, attributes=attributes or {})
                self._gauge_values[key] = value
        except Exception as e:
            logger.debug(f"Failed to set gauge {metric_name}: {e}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def measure_duration(
    metric_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    unit: str = "s",
):
    """
    Decorator for measuring function duration.
    
    Example:
        @measure_duration("actor.collect_trajectory.duration")
        def collect_trajectory(self):
            # function body
            pass
    
    Args:
        metric_name: Name of the duration metric
        attributes: Static attributes to add to all measurements
        unit: Unit of measurement (default: seconds)
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_telemetry_enabled():
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                
                try:
                    collector = get_metrics_collector()
                    attrs = dict(attributes) if attributes else {}
                    attrs["status"] = status
                    
                    # Try to add component info from self if available
                    if args and hasattr(args[0], "__class__"):
                        attrs.setdefault("component", args[0].__class__.__name__.lower())
                    
                    collector.record_duration(metric_name, duration, attrs, unit)
                except Exception as e:
                    logger.debug(f"Failed to record duration metric: {e}")
        
        return wrapper
    
    return decorator


def measure_async_duration(
    metric_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    unit: str = "s",
):
    """
    Decorator for measuring async function duration.
    
    Example:
        @measure_async_duration("actor.async_operation.duration")
        async def async_operation(self):
            # async function body
            pass
    
    Args:
        metric_name: Name of the duration metric
        attributes: Static attributes to add to all measurements
        unit: Unit of measurement (default: seconds)
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_telemetry_enabled():
                return await func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                
                try:
                    collector = get_metrics_collector()
                    attrs = dict(attributes) if attributes else {}
                    attrs["status"] = status
                    
                    # Try to add component info from self if available
                    if args and hasattr(args[0], "__class__"):
                        attrs.setdefault("component", args[0].__class__.__name__.lower())
                    
                    collector.record_duration(metric_name, duration, attrs, unit)
                except Exception as e:
                    logger.debug(f"Failed to record duration metric: {e}")
        
        return wrapper
    
    return decorator


def count_calls(
    metric_name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for counting function calls.
    
    Example:
        @count_calls("actor.trajectories_generated.total")
        def collect_trajectory(self):
            # function body
            pass
    
    Args:
        metric_name: Name of the counter metric
        attributes: Static attributes to add to all measurements
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                if is_telemetry_enabled():
                    collector = get_metrics_collector()
                    attrs = dict(attributes) if attributes else {}
                    attrs["status"] = "success"
                    collector.increment_counter(metric_name, 1, attrs)
                
                return result
            except Exception as e:
                if is_telemetry_enabled():
                    collector = get_metrics_collector()
                    attrs = dict(attributes) if attributes else {}
                    attrs["status"] = "error"
                    collector.increment_counter(metric_name, 1, attrs)
                raise
        
        return wrapper
    
    return decorator


def record_value(
    metric_name: str,
    value_func: Callable[[Any], Union[int, float]],
    attributes: Optional[Dict[str, Any]] = None,
    unit: str = "1",
):
    """
    Decorator for recording a value based on function result.
    
    Example:
        @record_value("actor.reward.mean", value_func=lambda result: result['mean_reward'])
        def score_molecules(self):
            # function body
            return {"mean_reward": 0.75}
    
    Args:
        metric_name: Name of the value metric
        value_func: Function to extract value from result
        attributes: Static attributes to add to all measurements
        unit: Unit of measurement
    """
    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if is_telemetry_enabled():
                try:
                    value = value_func(result)
                    collector = get_metrics_collector()
                    attrs = dict(attributes) if attributes else {}
                    collector.record_value(metric_name, value, attrs, unit)
                except Exception as e:
                    logger.debug(f"Failed to record value metric: {e}")
            
            return result
        
        return wrapper
    
    return decorator


# Convenience functions for direct metric recording
def record_duration_metric(
    metric_name: str,
    duration: float,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a duration metric directly."""
    if is_telemetry_enabled():
        get_metrics_collector().record_duration(metric_name, duration, attributes)


def record_value_metric(
    metric_name: str,
    value: Union[int, float],
    attributes: Optional[Dict[str, Any]] = None,
    unit: str = "1",
    snapshot: bool = False,
) -> None:
    """Record a value metric directly.
    
    Args:
        metric_name: Name of the metric
        value: Value to record
        attributes: Optional attributes for the metric
        unit: Unit of measurement
        snapshot: If True, record as absolute value (use for version numbers, etc.)
    """
    if is_telemetry_enabled():
        get_metrics_collector().record_value(metric_name, value, attributes, unit, snapshot)


def increment_counter_metric(
    metric_name: str,
    value: int = 1,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Increment a counter metric directly."""
    if is_telemetry_enabled():
        get_metrics_collector().increment_counter(metric_name, value, attributes)


def set_gauge_metric(
    metric_name: str,
    value: Union[int, float],
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Set a gauge metric directly."""
    if is_telemetry_enabled():
        get_metrics_collector().set_gauge(metric_name, value, attributes)
