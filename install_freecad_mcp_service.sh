#!/bin/bash

# FreeCAD MCP Service Installation Script (mirrors rhyme_mcp/install_rhyme_mcp_service.sh)
# Installs freecad_mcp.py (fc conda env) as systemd service freecad-mcp.
# The server starts its GPU helper (gpu_solver.py, py310_amd torch ROCm) on demand as a child process.
#
# Usage: sudo ./install_freecad_mcp_service.sh [--uninstall]
# Overrides via env: FREECAD_MCP_HOST, FREECAD_MCP_PORT, FREECAD_MCP_PATH, FC_ENV_PATH, GPU_PYTHON,
#                    FREECAD_MCP_DEVICE (auto|gpu|cpu), SERVICE_USER

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="freecad-mcp"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-wiffzack}}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
FC_ENV_PATH="${FC_ENV_PATH:-$SERVICE_HOME/miniforge3/envs/fc}"
PYTHON_PATH="$FC_ENV_PATH/bin/python"
GPU_PYTHON="${GPU_PYTHON:-/mnt/data/miniconda3/envs/py310_amd/bin/python}"
HOST="${FREECAD_MCP_HOST:-0.0.0.0}"
PORT="${FREECAD_MCP_PORT:-2022}"
URL_PATH="${FREECAD_MCP_PATH:-/freecad}"
DEVICE="${FREECAD_MCP_DEVICE:-auto}"
GLTFPACK_BIN="${GLTFPACK:-$(sudo -u "$SERVICE_USER" bash -lc 'command -v gltfpack' 2>/dev/null || echo "$SERVICE_HOME/.local/bin/gltfpack")}"

if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

if [[ "$1" == "--uninstall" ]]; then
    systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_FILE"
    systemctl daemon-reload
    echo -e "${GREEN}$SERVICE_NAME removed.${NC}"
    exit 0
fi

if [[ ! -x "$PYTHON_PATH" ]]; then
    echo -e "${RED}Python interpreter not found at $PYTHON_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Checking dependencies in the fc env...${NC}"
if ! sudo -u "$SERVICE_USER" HOME="$SERVICE_HOME" PYTHONNOUSERSITE=1 "$PYTHON_PATH" -c \
    "import sys; sys.path.append('$FC_ENV_PATH/lib'); import fastmcp, scipy, skimage, sksparse, optuna, FreeCAD" 2>/dev/null; then
    echo -e "${RED}Missing packages in $FC_ENV_PATH. Install with:${NC}"
    echo "  mamba install -n fc -c conda-forge calculix scipy scikit-image scikit-sparse 'lazy-loader<0.6'"
    echo "  $PYTHON_PATH -m pip install fastmcp trimesh optuna"
    exit 1
fi
if [[ "$DEVICE" != "cpu" ]]; then
    if sudo -u "$SERVICE_USER" HOME="$SERVICE_HOME" PYTHONNOUSERSITE=1 "$GPU_PYTHON" -c \
        "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${GREEN}GPU solver available via $GPU_PYTHON${NC}"
    else
        echo -e "${YELLOW}No GPU torch at $GPU_PYTHON - topology optimisation falls back to CPU${NC}"
    fi
fi

if [[ -x "$GLTFPACK_BIN" ]]; then
    echo -e "${GREEN}gltfpack: $GLTFPACK_BIN${NC}"
else
    echo -e "${YELLOW}gltfpack not found - .glb exports are skipped (npm install -g --prefix ~/.local gltfpack)${NC}"
fi

if ss -ltn "sport = :$PORT" | grep -q LISTEN && ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${RED}Port $PORT is already in use by another process (stop a manually started server first):${NC}"
    ss -ltnp "sport = :$PORT"
    exit 1
fi

echo -e "${YELLOW}Installing FreeCAD MCP as systemd service (user $SERVICE_USER)...${NC}"

cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=FreeCAD MCP Server (FEM + topology optimisation)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
# render/video give the GPU helper access to /dev/kfd and /dev/dri
SupplementaryGroups=render video
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_PATH $SCRIPT_DIR/freecad_mcp.py streamable-http
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

Environment="HOME=$SERVICE_HOME"
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONNOUSERSITE=1"
Environment="FREECAD_MCP_HOST=$HOST"
Environment="FREECAD_MCP_PORT=$PORT"
Environment="FREECAD_MCP_PATH=$URL_PATH"
Environment="FREECAD_MCP_DEVICE=$DEVICE"
Environment="FREECAD_GPU_PYTHON=$GPU_PYTHON"
Environment="GLTFPACK=$GLTFPACK_BIN"
Environment="KMP_AFFINITY=disabled"
Environment="KMP_WARNINGS=0"

NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT

chmod 644 "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo -e "${YELLOW}Waiting for port $PORT...${NC}"
for _ in $(seq 1 30); do
    ss -ltn "sport = :$PORT" | grep -q LISTEN && break
    sleep 1
done
systemctl status "$SERVICE_NAME" --no-pager || true

echo ""
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    echo -e "${GREEN}Installation complete!${NC}"
else
    echo -e "${RED}Service did not start listening on port $PORT - check the logs.${NC}"
fi
echo "  Logs:      journalctl -u $SERVICE_NAME -f"
echo "  Uninstall: sudo $0 --uninstall"
echo "Service is listening on: http://$HOST:$PORT$URL_PATH"
