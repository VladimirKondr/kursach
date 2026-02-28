# 📊 IMPALA RL OpenTelemetry Integration

Comprehensive observability for the IMPALA RL pipeline with distributed tracing, metrics, and structured logging.

## 🚀 Quick Start

### 1. Start SigNoz (Observability Backend)

```bash
cd examples
docker-compose -f docker-compose.signoz.yml up -d
```

SigNoz UI will be available at: http://localhost:3301

### 2. Initialize Telemetry in Your Code

```python
from reinvent.runmodes.impala_rl.telemetry import setup_telemetry

# Initialize telemetry (call once at startup)
setup_telemetry(env="dev")  # or "prod"
```

### 3. Run Your IMPALA Training

The instrumentation is already integrated into Actor, Learner, and NATS nodes.
Just run your training script and telemetry data will automatically flow to SigNoz.

## 📦 Installation

Install OpenTelemetry dependencies:

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc \
            opentelemetry-instrumentation \
            pyyaml pydantic
```

## 🔧 Configuration

### Environment-Based Configuration

Two pre-configured environments:

- **Dev** (`telemetry.dev.yaml`): Full sampling, DEBUG logs, localhost SigNoz
- **Prod** (`telemetry.prod.yaml`): 10% sampling, INFO logs, production endpoints

```python
# Use dev config
setup_telemetry(env="dev")

# Use prod config
setup_telemetry(env="prod")

# Use custom config file
setup_telemetry(config_path="/path/to/custom-config.yaml")
```

### Environment Variables Override

Standard OpenTelemetry environment variables are supported:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-signoz:4317"
export OTEL_SERVICE_NAME="reinvent-impala-prod"
export OTEL_TRACES_SAMPLER="trace_id_ratio"
export OTEL_TRACES_SAMPLER_ARG="0.1"
export IMPALA_TELEMETRY_ENABLED="true"
```

### Disable Telemetry

```bash
export IMPALA_TELEMETRY_ENABLED="false"
```

Or in code:

```python
# Telemetry will gracefully no-op if disabled
setup_telemetry(env="dev")  # Returns False if disabled
```

## 📈 What Gets Measured

### Traces (Distributed Tracing)

Full Actor → NATS → Learner distributed traces with parent-child relationships:

```
impala.training_iteration
├─ actor.collect_trajectory
│  ├─ actor.sampling
│  ├─ actor.scoring
│  └─ actor_node.send_trajectories
│     └─ nats.publish
├─ learner_node.get_trajectories
│  └─ nats.fetch
└─ learner.update
   ├─ learner.compute_target_log_probs
   ├─ learner.compute_importance_weights
   └─ learner_node.commit_model
```

### Metrics

**Performance Metrics:**
- `actor.trajectory_generation.duration` - Time to generate batch
- `actor.sampling.duration` - RNN sampling time
- `actor.scoring.duration` - Scoring function time
- `learner.update.duration` - Training step time
- `learner.batch_collection.duration` - Time to collect batch from NATS

**Quality Metrics:**
- `actor.reward.mean/std/max/min` - Reward statistics
- `actor.smiles.valid.ratio` - Valid SMILES percentage
- `learner.loss` - Training loss
- `learner.agent_nll.mean` - Agent negative log-likelihood
- `learner.importance_weights.mean/max` - Off-policy correction weights

**System Metrics:**
- `nats.queue.depth` - Pending messages
- `nats.publish.duration` - Message publish time
- `nats.message.size_bytes` - Message sizes
- `model.version.current` - Model version
- `model.version.lag` - Actor-Learner version difference

**Counters:**
- `actor.trajectories_generated.total`
- `learner.trajectories_processed.total`
- `nats.publish.failures.total`
- `nats.errors.total`

### Structured Logs

JSON-formatted logs with automatic trace context:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "level": "INFO",
  "message": "Actor progress",
  "service": "reinvent-impala",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "attributes": {
    "actor_id": 0,
    "steps": 100,
    "mean_reward": 0.75
  }
}
```

## 🎯 SigNoz Dashboards

### Viewing Data in SigNoz

1. **Traces**: Navigate to "Traces" tab
   - Filter by service: `reinvent-impala`
   - See end-to-end latency
   - Identify bottlenecks in flamegraph view

2. **Metrics**: Navigate to "Metrics" tab
   - Create custom queries
   - Example: `avg(actor.reward.mean) by actor_id`
   - Set up percentile views: p50, p95, p99

3. **Logs**: Navigate to "Logs" tab
   - Correlated with traces via trace_id
   - Filter by log level, component, attributes

### Example Queries

**Average trajectory generation time per actor:**
```
avg(actor.trajectory_generation.duration) by actor_id
```

**Training loss over time:**
```
learner.loss
```

**NATS queue depth:**
```
nats.queue.depth
```

**Success rate of trajectory fetches:**
```
(learner.trajectories_fetched.total - learner.get_trajectories.failures.total) / learner.trajectories_fetched.total * 100
```

## 🐳 Production Deployment

### Kubernetes Setup

1. Deploy SigNoz in your cluster (separate namespace):

```yaml
# Use SigNoz Helm chart or official k8s manifests
# https://signoz.io/docs/install/kubernetes/
```

2. Configure IMPALA pods with environment variables:

```yaml
env:
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://signoz-otel-collector.observability:4317"
  - name: OTEL_SERVICE_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name  # Pod name
  - name: K8S_POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: K8S_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
  - name: K8S_NODE_NAME
    valueFrom:
      fieldRef:
        fieldPath: spec.nodeName
  - name: IMPALA_ENV
    value: "prod"
```

3. Use production config:

```python
setup_telemetry(env="prod")
```

### Resource Attributes

In Kubernetes, telemetry automatically detects and adds:
- `k8s.pod.name`
- `k8s.namespace`
- `k8s.node.name`
- `host.name`
- `process.pid`

## 🔍 Troubleshooting

### Telemetry Not Working

1. **Check SigNoz is running:**
   ```bash
   curl http://localhost:4317
   ```

2. **Check telemetry initialization:**
   ```python
   from reinvent.runmodes.impala_rl.telemetry import is_telemetry_enabled
   print(is_telemetry_enabled())  # Should be True
   ```

3. **Check logs for errors:**
   - Look for "Telemetry initialization complete" message
   - Check for connection errors to SigNoz

### Data Not Appearing in SigNoz

1. **Verify OTLP collector is receiving data:**
   - Check `http://localhost:8888/metrics` (collector metrics endpoint)

2. **Check sampling:**
   - In production, only 10% of traces are sampled by default
   - Increase sampling ratio in config for testing

3. **Check time range in SigNoz UI:**
   - Make sure you're looking at the correct time window

### Performance Impact

- Overhead: < 5% latency increase
- Memory: < 100MB additional
- Disable if needed: `IMPALA_TELEMETRY_ENABLED=false`

## 📚 API Reference

### Setup

```python
setup_telemetry(
    config_path: Optional[str] = None,
    env: Optional[str] = None,
    test_connectivity: bool = True,
) -> bool
```

### Getting Instrumentation Objects

```python
from reinvent.runmodes.impala_rl.telemetry import get_tracer, get_meter, get_logger

tracer = get_tracer(__name__)
meter = get_meter(__name__)
logger = get_logger(__name__)
```

### Manual Instrumentation

```python
from reinvent.runmodes.impala_rl.telemetry.traces import create_span, add_span_attributes
from reinvent.runmodes.impala_rl.telemetry.metrics import record_value_metric
from reinvent.runmodes.impala_rl.telemetry.logs import log_context

# Create a span
with create_span("my_operation", tracer=tracer, attributes={"key": "value"}) as span:
    # Do work
    span.add_event("checkpoint")
    span.set_attribute("result", 42)

# Record a metric
record_value_metric("my_metric", 42, attributes={"component": "custom"})

# Structured logging with context
with log_context(actor_id=0, step=100):
    logger.info("Processing", attributes={"batch_size": 128})
```

## 🎓 Best Practices

1. **Always initialize telemetry at startup:**
   ```python
   setup_telemetry(env="dev")
   ```

2. **Use structured logging:**
   ```python
   logger.info("Message", attributes={"key": "value"})
   # NOT: logger.info(f"Message {key}={value}")
   ```

3. **Add meaningful span attributes:**
   ```python
   span.set_attribute("batch_size", 128)
   span.set_attribute("model_version", 5)
   ```

4. **Record errors properly:**
   ```python
   try:
       # code
   except Exception as e:
       span.record_exception(e)
       raise
   ```

5. **Use appropriate metrics:**
   - Durations → Histogram
   - Counts → Counter
   - Current values → Gauge

## 📞 Support

For issues or questions:
- Check SigNoz docs: https://signoz.io/docs/
- OpenTelemetry Python docs: https://opentelemetry.io/docs/languages/python/
- Review telemetry code: `reinvent/runmodes/impala_rl/telemetry/`

## 🔗 Links

- SigNoz: https://signoz.io
- OpenTelemetry: https://opentelemetry.io
- W3C Trace Context: https://www.w3.org/TR/trace-context/
