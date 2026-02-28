#!/usr/bin/env python3
"""
Fetch all logs/metrics for a specific IMPALA training run
"""

import subprocess
import sys
from datetime import datetime

def run_query(query):
    """Execute ClickHouse query"""
    try:
        result = subprocess.run(
            ["docker", "exec", "signoz-clickhouse", "clickhouse-client", "-q", query],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {e}"

def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "1772308951_ed9db67b"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "logs.txt"
    
    print(f"💾 Fetching data for run_id: {run_id}")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write(f"IMPALA TRAINING RUN LOGS - Run ID: {run_id}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 100 + "\n\n")
        
        # Fetch all metrics
        f.write("📊 ALL METRICS FOR THIS RUN\n")
        f.write("-" * 100 + "\n")
        
        query = f"""
        SELECT 
          toDateTime(unix_milli/1000) as timestamp,
          metric_name,
          value,
          attributes
        FROM signoz_metrics.distributed_samples_v4
        WHERE toDateTime(unix_milli/1000) >= now() - INTERVAL 2 HOUR
        ORDER BY timestamp
        FORMAT TabSeparatedWithNames
        """
        
        print("   Fetching metrics...")
        result = run_query(query)
        f.write(result)
        f.write("\n\n")
        
        # Summary statistics
        f.write("📈 SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n")
        
        # Get loss values
        query_loss = """
        SELECT 
          count() as count,
          min(value) as min_loss,
          max(value) as max_loss,
          avg(value) as avg_loss
        FROM signoz_metrics.distributed_samples_v4
        WHERE metric_name = 'learner.loss'
          AND toDateTime(unix_milli/1000) >= now() - INTERVAL 2 HOUR
        """
        
        print("   Calculating loss statistics...")
        result = run_query(query_loss)
        f.write(f"Loss Statistics:\n{result}\n\n")
        
        # Get reward values
        query_reward = """
        SELECT 
          count() as count,
          min(value) as min_reward,
          max(value) as max_reward,
          avg(value) as avg_reward
        FROM signoz_metrics.distributed_samples_v4
        WHERE metric_name = 'actor.reward.mean'
          AND toDateTime(unix_milli/1000) >= now() - INTERVAL 2 HOUR
        """
        
        print("   Calculating reward statistics...")
        result = run_query(query_reward)
        f.write(f"Reward Statistics:\n{result}\n\n")
        
        # Get validity
        query_validity = """
        SELECT 
          count() as count,
          min(value) as min_validity,
          max(value) as max_validity,
          avg(value) as avg_validity
        FROM signoz_metrics.distributed_samples_v4
        WHERE metric_name = 'actor.smiles.valid.ratio'
          AND toDateTime(unix_milli/1000) >= now() - INTERVAL 2 HOUR
        """
        
        print("   Calculating validity statistics...")
        result = run_query(query_validity)
        f.write(f"SMILES Validity Statistics:\n{result}\n\n")
        
        # List all unique metrics
        query_metrics = """
        SELECT 
          metric_name,
          count() as count
        FROM signoz_metrics.distributed_samples_v4
        WHERE toDateTime(unix_milli/1000) >= now() - INTERVAL 2 HOUR
        GROUP BY metric_name
        ORDER BY metric_name
        """
        
        print("   Listing all metrics...")
        result = run_query(query_metrics)
        f.write(f"All Metrics Collected:\n{result}\n\n")
        
    print(f"✅ Logs exported to {output_file}")
    
    # Show file size  
    import os
    size = os.path.getsize(output_file)
    print(f"   File size: {size:,} bytes ({size/1024:.1f} KB)")
    
    with open(output_file) as f:
        lines = len(f.readlines())
    print(f"   Lines: {lines:,}")

if __name__ == "__main__":
    main()
