# 🚀 Quick Start Guide: OpenTelemetry Integration

## Prerequisites

```bash
# Install OpenTelemetry dependencies
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc \
            opentelemetry-instrumentation \
            pyyaml pydantic
```

## 🎯 Quick Test (Recommended for First Time)

For quick testing without full SigNoz UI, use simple console output:

```bash
cd examples
docker-compose -f docker-compose.otel-simple.yml up -d
```

Then run IMPALA training:
```bash
python test_impala_full_integration.py
```

View telemetry in console:
```bash
docker logs -f otel-collector-simple
```

You'll see traces, metrics, and logs printed in real-time! ✅

---

## 🔥 Full Setup with SigNoz UI

For production-like setup with web UI and persistent storage:

## Step 1: Start SigNoz (Observability Backend)

```bash
cd examples
docker-compose -f docker-compose.signoz.yml up -d
```

**Note**: First startup takes ~60 seconds. Wait for all services to be healthy.

Check status:
```bash
docker ps
```

Wait until all containers show "(healthy)" status.

**Troubleshooting**: If SigNoz fails to start, use the simple setup above instead.

Open SigNoz UI: **http://localhost:3301**

## Step 2: Start NATS Server

```bash
# In a separate terminal
nats-server -js -D
```

## Step 3: Run IMPALA Training with Telemetry

```bash
# Telemetry is enabled by default
python test_impala_full_integration.py
```

## Step 4: View Observability Data

Open SigNoz UI at http://localhost:3301

### View Traces
- Navigate to **Traces** tab
- Select service: `reinvent-impala`
- Explore distributed traces: Actor → NATS → Learner

### View Metrics
- Navigate to **Metrics** tab
- Example queries:
  - `actor.reward.mean` - Average reward over time
  - `learner.loss` - Training loss
  - `nats.queue.depth` - Queue backlog

### View Logs
- Navigate to **Logs** tab
- Logs are automatically correlated with traces via `trace_id`
- Filter by component, actor_id, etc.

## Disable Telemetry (Optional)

```bash
export IMPALA_TELEMETRY_ENABLED=false
python test_impala_full_integration.py
```

## What You'll See

### Traces
Complete distributed traces showing:
- End-to-end latency (Actor → Learner)
- Time spent in each operation
- Bottleneck identification
- Parent-child span relationships

### Metrics
Real-time metrics:
- **Performance**: trajectory generation time, scoring time, training time
- **Quality**: rewards, loss, validity ratio
- **System**: queue depth, message sizes, connection states

### Logs
Structured JSON logs with:
- Trace context for correlation
- Actor/Learner lifecycle events
- Error tracking
- Performance insights

## ✅ How to Verify System is Working

### 1. Check Console Output

During test run, you should see:
```
✅ Telemetry initialized successfully
✅ NATS connection established
✅ Actor sampling molecules...
✅ Learner training...
📊 [Actor 0] Steps: 100 | Mean reward: 0.XXX
📉 [Learner] Step: 10 | Loss: X.XXX
```

### 2. Verify Model is Training

**Watch for these indicators:**

#### A. Loss Decreases Over Time
```bash
# In SigNoz Metrics tab, query:
learner.loss
```
✅ **Expected**: Loss should trend downward
❌ **Problem**: Loss is flat or increasing → Check learning rate, data pipeline

#### B. Rewards Increase (or stabilize)
```bash
# Query in SigNoz:
actor.reward.mean
```
✅ **Expected**: Rewards improve over iterations
❌ **Problem**: Rewards stuck at 0 → Check scoring function

#### C. Model Version Updates
```bash
# Query:
model.version.current
```
✅ **Expected**: Version number increases (0, 1, 2, 3...)
❌ **Problem**: Version stuck at 0 → Learner not committing updates

#### D. Actor-Learner Synchronization
```bash
# Query:
model.version.lag
```
✅ **Expected**: Lag < 10 (actors are reasonably up-to-date)
❌ **Problem**: Lag > 50 → NATS communication issue

### 3. System Health Checks

#### A. NATS Queue Depth
```bash
# Query:
nats.queue.depth
```
✅ **Expected**: Queue depth varies but doesn't grow unbounded
❌ **Problem**: Queue depth keeps increasing → Learner can't keep up

#### B. Message Flow
```bash
# Query:
nats.publish.total
nats.fetch.total
```
✅ **Expected**: Both metrics increase steadily
❌ **Problem**: Publish works but fetch=0 → Subscription issue

#### C. Validity Ratio
```bash
# Query:
actor.smiles.valid.ratio
```
✅ **Expected**: > 0.5 (at least 50% valid molecules)
❌ **Problem**: < 0.1 → Model generating invalid SMILES

### 4. Quick Validation Test

Run this after 5 minutes of training:

```python
# Check final metrics from console output
grep "Mean reward" test_output.log | tail -5
# Should show reward progression like:
# Mean reward: 0.123
# Mean reward: 0.156
# Mean reward: 0.189  ← increasing!
```

### 5. Trace Visualization

In SigNoz → Traces tab:
1. Find a trace from `reinvent-impala`
2. Expand the spans
3. ✅ **You should see**:
   ```
   actor.collect_trajectory (parent)
     ├─ sampling_molecules (child)
     ├─ scoring_molecules (child)
     └─ nats.publish (child)
   
   learner.update (parent)
     ├─ compute_v_trace (child)
     └─ backpropagation (child)
   ```

### 6. Compare Initial vs Final Model

```python
# After training completes, check model performance:

# Initial model (step 0):
# - Random rewards ~0.0-0.2
# - High loss ~5.0-10.0

# After 1000 steps:
# - Improved rewards ~0.4-0.8
# - Reduced loss ~1.0-2.0
```

### 7. Expected Timeline

| Time | Expected Behavior |
|------|-------------------|
| 0-30s | ✅ Telemetry + NATS connection established |
| 30-60s | ✅ First trajectories generated, Learner starts training |
| 1-2 min | ✅ First model update committed, Actors receive new version |
| 5-10 min | ✅ Noticeable reward improvement, Loss decreasing |
| 30+ min | ✅ Training converges, Metrics stabilize |

### 8. Debug Mode

For detailed debugging, enable verbose logging:

```bash
export IMPALA_LOG_LEVEL=DEBUG
export IMPALA_TELEMETRY_ENABLED=true
python test_impala_full_integration.py 2>&1 | tee training.log
```

Then inspect:
```bash
# Check Actor is sampling
grep "Actor progress" training.log

# Check Learner is training
grep "Training step" training.log

# Check model updates
grep "Model updated" training.log
```

## Troubleshooting

### SigNoz Not Starting
```bash
# Check docker containers
docker ps

# Check logs
docker-compose -f examples/docker-compose.signoz.yml logs
```

### Telemetry Not Working
Check the console output:
- ✅ "Telemetry initialized" - Working correctly
- ⚠️ "Telemetry fallback mode" - SigNoz not reachable, using console output
- ℹ️ "Telemetry disabled" - Explicitly disabled

### Data Not Appearing in SigNoz
1. Wait 30-60 seconds for data to appear
2. Check time range in SigNoz UI (top right corner)
3. Verify OTLP collector is running: `curl http://localhost:4317`

## Next Steps

- Read full documentation: [telemetry/README.md](reinvent/runmodes/impala_rl/telemetry/README.md)
- Customize dashboards in SigNoz
- Set up alerts for critical metrics
- Deploy to production (see production guide in README)

## Clean Up

```bash
# Stop SigNoz
cd examples
docker-compose -f docker-compose.signoz.yml down

# Remove data volumes (optional)
docker-compose -f docker-compose.signoz.yml down -v

# Stop NATS
# Press Ctrl+C in NATS terminal
```

## Key Endpoints

- SigNoz UI: http://localhost:3301
- OTLP gRPC: http://localhost:4317
- OTLP HTTP: http://localhost:4318
- Collector Metrics: http://localhost:8888/metrics
- ClickHouse: localhost:9000
- NATS: localhost:4222

## Architecture

```
┌─────────────┐      OTLP      ┌────────────────┐
│   Actor     │────────────────▶│                │
│   Learner   │                 │ OTEL Collector │
│   NATS      │                 │                │
└─────────────┘                 └────────┬───────┘
                                         │
                                         ▼
                                ┌────────────────┐
                                │  ClickHouse    │
                                │  (Storage)     │
                                └────────┬───────┘
                                         │
                                         ▼
                                ┌────────────────┐
                                │ SigNoz Query   │◀──── You view this
                                │ + Frontend     │      at :3301
                                └────────────────┘
```

## Example Queries in SigNoz

```sql
-- Average reward per actor
avg(actor.reward.mean) by actor_id

-- 95th percentile trajectory generation time
p95(actor.trajectory_generation.duration)

-- NATS queue depth over time
nats.queue.depth

-- Success rate
(actor.trajectories_generated.total / (actor.trajectories_generated.total + actor.errors.total)) * 100
```

## Support

For issues:
- Check logs: `docker-compose -f examples/docker-compose.signoz.yml logs`
- SigNoz docs: https://signoz.io/docs/
- OpenTelemetry docs: https://opentelemetry.io/docs/languages/python/
