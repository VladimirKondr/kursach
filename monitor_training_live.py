#!/usr/bin/env python3
"""
Real-time IMPALA Training Monitor
Shows key metrics in terminal with live updates
"""

import time
import sys
from datetime import datetime, timedelta
import subprocess

def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")

def get_metric_value(metric_name, time_window_minutes=5):
    """Get latest value for a metric from ClickHouse"""
    query = f"""
    SELECT value 
    FROM distributed_samples_v4 
    WHERE metric_name = '{metric_name}' 
      AND unix_milli >= (toUnixTimestamp(now()) - {time_window_minutes * 60}) * 1000
    ORDER BY unix_milli DESC 
    LIMIT 1
    """
    
    try:
        result = subprocess.run(
            ["docker", "exec", "signoz-clickhouse", "clickhouse-client", "--query", query],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return None
    except Exception:
        return None

def get_metric_stats(metric_name, time_window_minutes=5):
    """Get avg/min/max for a metric"""
    query = f"""
    SELECT 
        avg(value) as avg_val,
        min(value) as min_val,
        max(value) as max_val,
        count() as count
    FROM distributed_samples_v4 
    WHERE metric_name = '{metric_name}' 
      AND unix_milli >= (toUnixTimestamp(now()) - {time_window_minutes * 60}) * 1000
    """
    
    try:
        result = subprocess.run(
            ["docker", "exec", "signoz-clickhouse", "clickhouse-client", "--query", query],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split('\t')
            if len(parts) >= 4:
                return {
                    'avg': float(parts[0]),
                    'min': float(parts[1]),
                    'max': float(parts[2]),
                    'count': int(parts[3])
                }
        return None
    except Exception:
        return None

def get_per_actor_metric(metric_name, time_window_minutes=5):
    """Get metric values grouped by actor_id"""
    query = f"""
    SELECT 
        attributes['actor_id'] as actor_id,
        avg(value) as avg_val
    FROM distributed_samples_v4 
    WHERE metric_name = '{metric_name}' 
      AND unix_milli >= (toUnixTimestamp(now()) - {time_window_minutes * 60}) * 1000
    GROUP BY actor_id
    ORDER BY actor_id
    """
    
    try:
        result = subprocess.run(
            ["docker", "exec", "signoz-clickhouse", "clickhouse-client", "--query", query],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout.strip():
            actors = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    actors[parts[0]] = float(parts[1])
            return actors
        return {}
    except Exception:
        return {}

def format_value(value, decimals=4):
    """Format value with color"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def color_text(text, color):
    """Color terminal text"""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"

def print_dashboard():
    """Print dashboard with current metrics"""
    clear_screen()
    
    print(color_text("=" * 80, 'bold'))
    print(color_text("🚀 IMPALA TRAINING MONITOR - LIVE DASHBOARD", 'bold'))
    print(color_text(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'white'))
    print(color_text("=" * 80, 'bold'))
    print()
    
    # Training Loss
    print(color_text("🎯 TRAINING LOSS (Lower is Better)", 'bold'))
    loss = get_metric_value('learner.loss')
    if loss:
        color = 'green' if loss < 5000 else 'yellow' if loss < 10000 else 'red'
        print(f"   Loss: {color_text(format_value(loss), color)}")
    else:
        print(f"   Loss: {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Rewards
    print(color_text("🎁 REWARDS (Higher is Better)", 'bold'))
    reward_stats = get_metric_stats('actor.reward.mean')
    if reward_stats and reward_stats['count'] > 0:
        print(f"   Mean:  {color_text(format_value(reward_stats['avg']), 'green')}")
        print(f"   Min:   {format_value(reward_stats['min'])}")
        print(f"   Max:   {format_value(reward_stats['max'])}")
        print(f"   Samples: {reward_stats['count']}")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Validity
    print(color_text("✅ SMILES VALIDITY RATIO", 'bold'))
    validity_stats = get_metric_stats('actor.smiles.valid.ratio')
    if validity_stats and validity_stats['count'] > 0:
        avg_pct = validity_stats['avg'] * 100
        color = 'green' if avg_pct > 95 else 'yellow' if avg_pct > 90 else 'red'
        print(f"   Average: {color_text(f'{avg_pct:.2f}%', color)}")
        print(f"   Min: {validity_stats['min'] * 100:.2f}%")
        print(f"   Max: {validity_stats['max'] * 100:.2f}%")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Importance Weights
    print(color_text("⚖️  IMPORTANCE WEIGHTS", 'bold'))
    iw_mean = get_metric_value('learner.importance_weights.mean')
    iw_min = get_metric_value('learner.importance_weights.min')
    iw_max = get_metric_value('learner.importance_weights.max')
    
    if iw_mean:
        print(f"   Mean:  {format_value(iw_mean)} (should be ~1.0)")
        if iw_min:
            color = 'green' if iw_min > 0.1 else 'yellow' if iw_min > 0.05 else 'red'
            print(f"   Min:   {color_text(format_value(iw_min), color)} {'⚠️  LOW!' if iw_min < 0.1 else ''}")
        if iw_max:
            print(f"   Max:   {format_value(iw_max)}")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Per-Actor Importance Weights
    print(color_text("👥 PER-ACTOR IMPORTANCE WEIGHTS (Detect Lagging Actors)", 'bold'))
    actor_weights = get_per_actor_metric('learner.importance_weight')
    if actor_weights:
        for actor_id, weight in sorted(actor_weights.items()):
            color = 'green' if weight > 0.1 else 'yellow' if weight > 0.05 else 'red'
            status = '✅' if weight > 0.1 else '⚠️'
            print(f"   Actor {actor_id}: {color_text(format_value(weight), color)} {status}")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Model Version Lag
    print(color_text("🔄 MODEL VERSION LAG", 'bold'))
    lag_by_actor = get_per_actor_metric('model.version.lag')
    if lag_by_actor:
        for actor_id, lag in sorted(lag_by_actor.items()):
            color = 'green' if lag < 5 else 'yellow' if lag < 10 else 'red'
            status = '✅' if lag < 5 else '⚠️' if lag < 10 else '❌'
            print(f"   Actor {actor_id}: {color_text(format_value(lag, 0), color)} versions behind {status}")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Queue Depth
    print(color_text("📦 NATS QUEUE DEPTH", 'bold'))
    queue_depth = get_metric_value('nats.queue.depth')
    if queue_depth is not None:
        color = 'green' if queue_depth < 200 else 'yellow' if queue_depth < 500 else 'red'
        status = '✅ Healthy' if queue_depth < 200 else '⚠️  Growing' if queue_depth < 500 else '❌ Bottleneck!'
        print(f"   Depth: {color_text(format_value(queue_depth, 0), color)} messages {status}")
    else:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    # Durations
    print(color_text("⏱️  OPERATION DURATIONS (seconds)", 'bold'))
    learner_duration = get_metric_value('learner.update.duration')
    actor_duration = get_metric_value('actor.trajectory_generation.duration')
    
    if learner_duration:
        print(f"   Learner Update: {format_value(learner_duration, 3)}s")
    if actor_duration:
        print(f"   Actor Trajectory Gen: {format_value(actor_duration, 3)}s")
    
    if not learner_duration and not actor_duration:
        print(f"   {color_text('N/A (waiting for data...)', 'yellow')}")
    print()
    
    print(color_text("=" * 80, 'bold'))
    print(color_text("Press Ctrl+C to exit | Auto-refresh every 5 seconds", 'cyan'))
    print(color_text("=" * 80, 'bold'))

def main():
    """Main loop"""
    print(color_text("\n🚀 Starting IMPALA Training Monitor...\n", 'bold'))
    print("Connecting to SigNoz ClickHouse...")
    
    # Test connection
    try:
        subprocess.run(
            ["docker", "exec", "signoz-clickhouse", "clickhouse-client", "--query", "SELECT 1"],
            capture_output=True,
            timeout=2,
            check=True
        )
        print(color_text("✅ Connected to ClickHouse\n", 'green'))
    except subprocess.CalledProcessError:
        print(color_text("❌ Failed to connect to ClickHouse", 'red'))
        print("Make sure SigNoz is running: docker ps | grep signoz")
        sys.exit(1)
    except FileNotFoundError:
        print(color_text("❌ Docker not found", 'red'))
        sys.exit(1)
    
    time.sleep(2)
    
    try:
        while True:
            print_dashboard()
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print(color_text("\n\n👋 Monitor stopped by user\n", 'yellow'))
        sys.exit(0)

if __name__ == "__main__":
    main()
