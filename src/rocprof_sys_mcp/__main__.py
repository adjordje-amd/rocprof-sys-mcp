"""Entry point for the rocprof-sys-mcp server."""

import sys


def main():
    """Run the MCP server."""
    from rocprof_sys_mcp.server import run_server

    run_server()


if __name__ == "__main__":
    sys.exit(main() or 0)
