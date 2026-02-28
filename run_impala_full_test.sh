#!/bin/bash
# Full IMPALA Integration Test Runner
# Starts NATS server and runs complete training test

set -e

echo "============================================================"
echo "IMPALA Full Integration Test Runner"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if NATS is already running
if pgrep -x "nats-server" > /dev/null; then
    echo -e "${GREEN}✅ NATS server is already running${NC}"
    NATS_RUNNING=true
else
    echo -e "${YELLOW}⚠️  NATS server is not running${NC}"
    NATS_RUNNING=false
fi

# Function to cleanup
cleanup() {
    if [ "$NATS_RUNNING" = false ]; then
        echo ""
        echo "🧹 Stopping NATS server..."
        pkill -f nats-server || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Start NATS if not running
if [ "$NATS_RUNNING" = false ]; then
    echo "🚀 Starting NATS server with JetStream..."
    nats-server --jetstream -D &
    NATS_PID=$!
    sleep 2
    
    if pgrep -x "nats-server" > /dev/null; then
        echo -e "${GREEN}✅ NATS server started (PID: $NATS_PID)${NC}"
    else
        echo -e "${RED}❌ Failed to start NATS server${NC}"
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "reinvent.prior" ]; then
    echo -e "${RED}❌ Model file 'reinvent.prior' not found${NC}"
    echo "   Please provide a trained REINVENT model"
    exit 1
fi

# Run the full integration test
echo ""
echo "============================================================"
echo "🧪 Running IMPALA Full Integration Test..."
echo "============================================================"
echo ""

python test_impala_full_integration.py

TEST_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ IMPALA FULL INTEGRATION TEST PASSED${NC}"
else
    echo -e "${RED}❌ IMPALA FULL INTEGRATION TEST FAILED${NC}"
fi
echo "============================================================"

exit $TEST_EXIT_CODE
