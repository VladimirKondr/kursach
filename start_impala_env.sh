#!/bin/bash
# Start full IMPALA environment: NATS + Actor test

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting IMPALA Environment"
echo "=========================================="

# Check if nats-server exists
if ! command -v nats-server &> /dev/null; then
    echo "❌ nats-server not found"
    echo "Install with: brew install nats-server"
    exit 1
fi

# Start NATS in background
echo ""
echo "1. Starting NATS JetStream server..."
nats-server --jetstream -D &
NATS_PID=$!
echo "   NATS PID: $NATS_PID"

# Wait for NATS to start
sleep 2

# Check if NATS is running
if ! nc -z localhost 4222 2>/dev/null; then
    echo "❌ NATS failed to start"
    kill $NATS_PID 2>/dev/null || true
    exit 1
fi

echo "✓ NATS is running on localhost:4222"

# Trap to kill NATS on exit
trap "kill $NATS_PID 2>/dev/null || true" EXIT

# Run test
echo ""
echo "2. Running Actor test..."
cd "$PROJECT_ROOT"

if python3 test_actor_simple.py; then
    echo ""
    echo "=========================================="
    echo "✓ Environment test passed!"
    echo "=========================================="
    echo ""
    echo "To run custom training:"
    echo "1. Keep NATS running in this terminal"
    echo "2. In another terminal:"
    echo "   cd $PROJECT_ROOT"
    echo "   python3 your_impala_script.py"
    echo ""
    
    # Keep running
    echo "Keeping NATS server running... (Press Ctrl+C to stop)"
    wait $NATS_PID
else
    echo ""
    echo "❌ Test failed"
    exit 1
fi
