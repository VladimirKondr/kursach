#!/bin/bash
# Fetch logs/traces for run_id

RUN_ID="${1:-1772308951_ed9db67b}"
OUTPUT="${2:-logs.txt}"

echo "Fetching data for run_id: $RUN_ID"
echo "=================================================================================" > "$OUTPUT"
echo "IMPALA Training Run Logs - Run ID: $RUN_ID" >> "$OUTPUT"
echo "Generated: $(date)" >> "$OUTPUT"
echo "=================================================================================" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Fetch metrics
echo "📊 METRICS" >> "$OUTPUT"
echo "--------------------------------------------------------------------------------" >> "$OUTPUT"

docker exec signoz-clickhouse clickhouse-client -q "
SELECT 
  toDateTime(unix_milli/1000) as time,
  metric_name,
  value,
  attributes
FROM signoz_metrics.distributed_samples_v4
WHERE attributes['run_id'] = '$RUN_ID'
ORDER BY time
FORMAT TabSeparated
" 2>/dev/null >> "$OUTPUT"

echo "" >> "$OUTPUT"
echo "✅ Logs exported to $OUTPUT"
wc -l "$OUTPUT"
