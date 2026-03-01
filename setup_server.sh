#!/usr/bin/env bash
# =============================================================================
# setup_server.sh — Full server bootstrap for the kursach project
# Usage: bash setup_server.sh
# Tested on Ubuntu 22.04 / 24.04
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/VladimirKondr/kursach.git"
SIGNOZ_REPO_URL="https://github.com/SigNoz/signoz.git"
SIGNOZ_VERSION="v0.88.0"  # pin a stable release; update as needed
KURSACH_DIR="$HOME/kursach"
SIGNOZ_DIR="$HOME/signoz"
PYTHON_VERSION="3.10"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $*${NC}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $*${NC}" >&2; exit 1; }

# =============================================================================
# 0. Detect OS (only Debian/Ubuntu supported)
# =============================================================================
if ! command -v apt-get &>/dev/null; then
    die "This script requires a Debian/Ubuntu system with apt-get."
fi

log "Updating package lists..."
sudo apt-get update -qq

# =============================================================================
# 1. Clone the kursach repository
# =============================================================================
log "=== Step 1: Cloning kursach repository ==="
if [ -d "$KURSACH_DIR/.git" ]; then
    warn "Repository already exists at $KURSACH_DIR — pulling latest changes."
    git -C "$KURSACH_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$KURSACH_DIR"
fi
log "Repository ready at $KURSACH_DIR"

# =============================================================================
# 2. Install Python 3.10
# =============================================================================
log "=== Step 2: Installing Python $PYTHON_VERSION ==="
if python3.10 --version &>/dev/null 2>&1; then
    warn "Python 3.10 already installed: $(python3.10 --version)"
else
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y \
        python3.10 python3.10-venv python3.10-dev python3.10-distutils \
        python3-pip build-essential libssl-dev libffi-dev
fi

# Make pip available for 3.10
if ! python3.10 -m pip --version &>/dev/null 2>&1; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
fi
log "Python 3.10: $(python3.10 --version)"

# =============================================================================
# 3. Install Docker + Docker Compose plugin
# =============================================================================
log "=== Step 3: Installing Docker ==="
if docker version &>/dev/null 2>&1; then
    warn "Docker already installed: $(docker version --format '{{.Server.Version}}')"
else
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                            docker-buildx-plugin docker-compose-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    warn "Added $USER to the docker group. You may need to 're-login' or run:"
    warn "  newgrp docker"
fi

# Allow current shell session to use docker without re-login
if ! groups | grep -q docker; then
    newgrp docker <<INNERSCRIPT
    echo "Switched to docker group inside subshell."
INNERSCRIPT
fi

log "Docker: $(sudo docker version --format '{{.Server.Version}}' 2>/dev/null || docker version --format '{{.Server.Version}}')"

# =============================================================================
# 4. Install Python pip dependencies
# =============================================================================
log "=== Step 4: Installing pip dependencies ==="

# System libs required by some packages (rdkit, opencv, etc.)
sudo apt-get install -y \
    libxrender1 libxext6 libsm6 \
    libgl1-mesa-glx libglib2.0-0 \
    libgomp1 || true

# Create a requirements file (from kursach conda env snapshot)
REQUIREMENTS_FILE="$KURSACH_DIR/requirements_server.txt"
cat > "$REQUIREMENTS_FILE" << 'EOF'
# Auto-generated from kursach conda environment (2026-03-01)
# ---- Core scientific stack ----
numpy==1.26.4
pandas==2.3.3
scipy==1.15.3
scikit-learn==1.7.2
matplotlib==3.10.8

# ---- Chemistry ----
rdkit==2025.9.4
chemprop==1.5.2
descriptastorus==2.8.0
MolVS==0.1.1
mmpdb==2.1

# ---- Deep learning (CPU build) — installed separately below ----
# torch and torchvision are pinned to the CPU wheel index

# ---- ML utilities ----
tensorboard==2.20.0
tensorboardX==2.6.4
hyperopt==0.2.7

# ---- Telemetry / observability ----
opentelemetry-api==1.39.1
opentelemetry-sdk==1.39.1
opentelemetry-exporter-otlp-proto-grpc==1.39.1
opentelemetry-exporter-otlp-proto-common==1.39.1
opentelemetry-instrumentation==0.60b1
opentelemetry-proto==1.39.1
opentelemetry-semantic-conventions==0.60b1
grpcio==1.78.0
googleapis-common-protos==1.72.0
protobuf==6.33.5

# ---- NATS messaging ----
nats-py==2.13.1

# ---- Web / utilities ----
Flask==3.1.2
requests==2.32.5
pydantic==2.12.5
PyYAML==6.0.3
click==8.3.1
tqdm==4.67.3
joblib==1.5.0
tenacity==8.5.0
python-dotenv==1.2.1
filelock==3.20.0
fsspec==2025.12.0
cloudpickle==3.1.2
sympy==1.14.0
networkx==3.4.2
polars==1.38.1
xarray==2025.6.1

# ---- Testing ----
pytest==8.4.2
pytest-mock==3.15.1
requests-mock==1.12.1

# ---- Other ----
cowsay==6.1
appdirs==1.4.4
arrow==1.4.0
typed-argument-parser==1.11.0
pumas==1.3.0
peewee==3.19.0
pandas-flavor==0.8.1
py4j==0.10.9.9
opencv-contrib-python-headless==4.12.0.88
pillow==10.4.0
xxhash==3.6.0
EOF

log "Installing PyTorch CPU wheels..."
python3.10 -m pip install --upgrade pip setuptools wheel
python3.10 -m pip install \
    torch==2.10.0+cpu torchvision==0.25.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

log "Installing remaining dependencies..."
python3.10 -m pip install -r "$REQUIREMENTS_FILE"

log "All pip packages installed."

# =============================================================================
# 5. Clone SigNoz and start the Docker stack
# =============================================================================
log "=== Step 5: Setting up SigNoz ==="

if [ -d "$SIGNOZ_DIR/.git" ]; then
    warn "SigNoz repo already exists at $SIGNOZ_DIR — pulling latest."
    git -C "$SIGNOZ_DIR" fetch --tags
    git -C "$SIGNOZ_DIR" checkout "$SIGNOZ_VERSION" 2>/dev/null || git -C "$SIGNOZ_DIR" pull
else
    git clone --depth 1 --branch "$SIGNOZ_VERSION" "$SIGNOZ_REPO_URL" "$SIGNOZ_DIR"
fi

SIGNOZ_COMPOSE="$SIGNOZ_DIR/deploy/docker/docker-compose.yaml"
if [ ! -f "$SIGNOZ_COMPOSE" ]; then
    die "SigNoz docker-compose.yaml not found at expected path: $SIGNOZ_COMPOSE"
fi

log "Starting SigNoz containers (this may take several minutes on first run)..."
cd "$SIGNOZ_DIR/deploy/docker"

DOCKER_CMD="docker"
# Re-use sudo if the user is not yet in docker group in this session
if ! $DOCKER_CMD info &>/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
fi

$DOCKER_CMD compose -f docker-compose.yaml up -d --remove-orphans

# =============================================================================
# 6. Wait for SigNoz to become healthy (query-service on :8080)
# =============================================================================
log "=== Step 6: Waiting for SigNoz to be ready ==="
HEALTH_URL="http://localhost:8080/api/v1/health"
MAX_WAIT=300   # seconds
INTERVAL=10
elapsed=0

until curl -sf "$HEALTH_URL" | grep -q '"status":"ok"' 2>/dev/null; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        warn "SigNoz did not become healthy within ${MAX_WAIT}s."
        warn "Check logs with: docker compose -f $SIGNOZ_COMPOSE logs -f"
        break
    fi
    echo -n "."
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done
echo ""
log "SigNoz is up and responding at $HEALTH_URL"

# =============================================================================
# 7. Open port 8080 in the firewall
# =============================================================================
log "=== Step 7: Exposing port 8080 ==="

# UFW (Ubuntu default)
if command -v ufw &>/dev/null; then
    sudo ufw allow 8080/tcp comment "SigNoz query-service" || true
    log "UFW rule added for port 8080/tcp"
fi

# iptables fallback (also covers servers without ufw)
if sudo iptables -C INPUT -p tcp --dport 8080 -j ACCEPT &>/dev/null 2>&1; then
    warn "iptables rule for port 8080 already exists."
else
    sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
    log "iptables rule added to allow inbound TCP 8080"
fi

# Persist iptables rules across reboots (if the package is available)
if dpkg -l iptables-persistent &>/dev/null 2>&1; then
    sudo netfilter-persistent save || true
else
    sudo apt-get install -y iptables-persistent <<< $'yes\nyes' || true
    sudo netfilter-persistent save 2>/dev/null || true
fi

# =============================================================================
# Done
# =============================================================================
SERVER_IP=$(curl -s https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN} Setup complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  kursach repo    : $KURSACH_DIR"
echo "  SigNoz dir      : $SIGNOZ_DIR"
echo ""
echo "  SigNoz UI       : http://${SERVER_IP}:3301"
echo "  SigNoz API      : http://${SERVER_IP}:8080"
echo ""
echo "  For SSH-tunnel access from your local machine:"
echo "    ssh -L 3301:localhost:3301 -L 8080:localhost:8080 <user>@${SERVER_IP}"
echo "  Then open http://localhost:3301 in your browser."
echo ""
echo "  To restart SigNoz:"
echo "    cd $SIGNOZ_DIR/deploy/docker && docker compose up -d"
echo ""
echo "  To view SigNoz logs:"
echo "    cd $SIGNOZ_DIR/deploy/docker && docker compose logs -f"
echo ""
