#!/bin/bash
# Quick test runner for IMPALA

set -e  # Exit on error

echo "=========================================="
echo "IMPALA Integration Test Suite"
echo "=========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "\n${BLUE}1. Checking prerequisites...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found"
    exit 1
fi

# Check NATS
if ! command -v nats-server &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} nats-server not found"
    echo "  Install with: brew install nats-server (macOS)"
    echo "  Or download from: https://github.com/nats-io/nats-server/releases"
else
    NATS_VERSION=$(nats-server --version 2>&1 | head -1)
    echo -e "${GREEN}✓${NC} NATS: $NATS_VERSION"
fi

# Check if NATS is running
echo ""
echo -e "\n${BLUE}2. Checking NATS server...${NC}"

if nc -z localhost 4222 2>/dev/null; then
    echo -e "${GREEN}✓${NC} NATS is running on localhost:4222"
else
    echo -e "${YELLOW}⚠${NC} NATS is NOT running"
    echo ""
    echo "Start NATS in another terminal:"
    echo "  nats-server -D"
    echo ""
    echo "Or start NATS with JetStream:"
    echo "  nats-server --jetstream -D"
    echo ""
    echo "Continuing with tests that don't require NATS..."
fi

# Check models
echo ""
echo -e "\n${BLUE}3. Checking models...${NC}"

if [ -f "$PROJECT_ROOT/reinvent.prior" ]; then
    echo -e "${GREEN}✓${NC} Found: reinvent.prior"
elif [ -f "$PROJECT_ROOT/examples/reinvent.prior" ]; then
    echo -e "${GREEN}✓${NC} Found: examples/reinvent.prior"
else
    echo -e "${YELLOW}⚠${NC} No REINVENT model found"
    echo "  Expected: reinvent.prior or examples/reinvent.prior"
    echo "  Download or train a model first"
fi

# Run simple actor test
echo ""
echo -e "\n${BLUE}4. Running Actor Test...${NC}"
echo "Command: python3 test_actor_simple.py"
echo ""

if python3 test_actor_simple.py; then
    echo -e "\n${GREEN}✓ Actor Test PASSED${NC}"
else
    echo -e "\n${RED}✗ Actor Test FAILED${NC}"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ All tests passed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If NATS wasn't running, start it: nats-server -D"
echo "2. Run full integration test: python3 test_impala_integration.py"
echo ""
