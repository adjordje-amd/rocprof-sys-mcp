#!/bin/bash

# rocprof-sys-mcp Installation Script
# This script sets up a Python virtual environment, installs dependencies,
# and configures the MCP server in ~/.claude.json

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path of the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo -e "${BLUE}=== rocprof-sys-mcp Installation ===${NC}"
echo ""

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${RED}Error: Python ${REQUIRED_VERSION}+ is required. Found: ${PYTHON_VERSION}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
echo ""

# Check for rocprofiler-systems
echo -e "${BLUE}Checking for rocprofiler-systems...${NC}"
if command -v rocprof-sys-run &> /dev/null; then
    ROCPROF_VERSION=$(rocprof-sys-run --version 2>&1 | head -n 1 || echo "unknown")
    echo -e "${GREEN}✓ rocprofiler-systems found: ${ROCPROF_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠ Warning: rocprofiler-systems not found in PATH${NC}"
    echo -e "${YELLOW}  GPU profiling will not work without it${NC}"
fi
echo ""

# Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing virtual environment...${NC}"
        rm -rf "${VENV_DIR}"
    else
        echo -e "${BLUE}Using existing virtual environment${NC}"
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo -e "${BLUE}Creating virtual environment at ${VENV_DIR}...${NC}"
    python3 -m venv "${VENV_DIR}"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment and install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install the package in editable mode from project root
pip install -e "${PROJECT_ROOT}"

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Configure Claude MCP
echo -e "${BLUE}Configuring MCP server...${NC}"
echo ""

# Check if config path provided as argument
if [ -n "$1" ]; then
    CLAUDE_CONFIG="$1"
else
    echo -e "Enter the path to your Claude config file"
    echo -e "(Press Enter to use default: ${HOME}/.claude.json):"
    read -r CLAUDE_CONFIG_INPUT

    if [ -z "${CLAUDE_CONFIG_INPUT}" ]; then
        CLAUDE_CONFIG="${HOME}/.claude.json"
    else
        CLAUDE_CONFIG="${CLAUDE_CONFIG_INPUT}"
    fi
fi

# Expand ~ to home directory if present
CLAUDE_CONFIG="${CLAUDE_CONFIG/#\~/$HOME}"

echo -e "${BLUE}Using config file: ${CLAUDE_CONFIG}${NC}"
echo ""

# Create directory if it doesn't exist
CLAUDE_CONFIG_DIR=$(dirname "${CLAUDE_CONFIG}")
if [ ! -d "${CLAUDE_CONFIG_DIR}" ]; then
    echo -e "${BLUE}Creating directory ${CLAUDE_CONFIG_DIR}${NC}"
    mkdir -p "${CLAUDE_CONFIG_DIR}"
fi

# Backup existing config if it exists
if [ -f "${CLAUDE_CONFIG}" ]; then
    BACKUP_FILE="${CLAUDE_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}Backing up existing config to ${BACKUP_FILE}${NC}"
    cp "${CLAUDE_CONFIG}" "${BACKUP_FILE}"
fi

# Create or update the Claude config
if [ ! -f "${CLAUDE_CONFIG}" ]; then
    # Create new config file
    echo -e "${BLUE}Creating new Claude config file${NC}"
    cat > "${CLAUDE_CONFIG}" << EOF
{
  "mcpServers": {
    "rocprof-sys": {
      "command": "${PYTHON_BIN}",
      "args": ["-m", "rocprof_sys_mcp"],
      "cwd": "${PROJECT_ROOT}"
    }
  }
}
EOF
else
    # Update existing config using Python
    echo -e "${BLUE}Updating existing Claude config file${NC}"
    python3 << EOF
import json
import sys

config_file = "${CLAUDE_CONFIG}"

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except json.JSONDecodeError:
    print("Error: Invalid JSON in ${CLAUDE_CONFIG}", file=sys.stderr)
    sys.exit(1)

# Ensure mcpServers key exists
if "mcpServers" not in config:
    config["mcpServers"] = {}

# Add or update rocprof-sys server
config["mcpServers"]["rocprof-sys"] = {
    "command": "${PYTHON_BIN}",
    "args": ["-m", "rocprof_sys_mcp"],
    "cwd": "${PROJECT_ROOT}"
}

# Write back to file
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
EOF
fi

echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# Test the installation
echo -e "${BLUE}Testing MCP server...${NC}"
if "${PYTHON_BIN}" -c "import rocprof_sys_mcp; print('Import successful')" &> /dev/null; then
    echo -e "${GREEN}✓ MCP server can be imported successfully${NC}"
else
    echo -e "${RED}✗ Error importing MCP server${NC}"
    exit 1
fi
echo ""

# Print summary
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo -e "Virtual environment: ${VENV_DIR}"
echo -e "Python binary: ${PYTHON_BIN}"
echo -e "Claude config: ${CLAUDE_CONFIG}"
echo ""
echo -e "${BLUE}To manually activate the virtual environment:${NC}"
echo -e "  source ${VENV_DIR}/bin/activate"
echo ""
echo -e "${BLUE}To test the server manually:${NC}"
echo -e "  ${PYTHON_BIN} -m rocprof_sys_mcp"
echo ""
echo -e "${GREEN}The MCP server is now configured and ready to use with Claude!${NC}"
