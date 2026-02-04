# rocprof-sys-mcp

MCP server for rocprofiler-systems profiling integration. Enables AI assistants to run profiling, analyze Perfetto traces, and generate performance reports.

## Requirements

- Python 3.10+
- rocprofiler-systems installed and in PATH
- AMD GPU with ROCm support (for GPU profiling)

## Installation

```bash
pip install -e .
```

## Usage

### As MCP Server

```bash
python -m rocprof_sys_mcp
```

### MCP Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "rocprof-sys": {
      "command": "python",
      "args": ["-m", "rocprof_sys_mcp"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `profile_application` | Run rocprof-sys-run on an application |
| `load_trace` | Load existing Perfetto trace file |
| `query_trace` | Execute SQL query on trace data |
| `get_trace_summary` | Get high-level trace statistics |
| `get_gpu_kernel_stats` | GPU kernel performance analysis |
| `generate_report` | Create markdown/JSON performance report |
| `compare_traces` | Compare two traces for regressions |
| `list_traces` | List loaded traces in session |

## Available Resources

| Resource URI | Description |
|--------------|-------------|
| `trace://{id}/metadata` | Trace metadata (system info, config) |
| `trace://{id}/schema` | Available tables and columns |
| `trace://active/info` | Currently active trace info |
| `session://traces` | All loaded traces |

## Example Workflow

1. Profile an application:
   ```
   profile_application(command="./my_app", args=["--input", "data.bin"])
   ```

2. Query the trace:
   ```
   query_trace(sql="SELECT name, dur FROM slice ORDER BY dur DESC LIMIT 10")
   ```

3. Generate a report:
   ```
   generate_report(format="markdown")
   ```

## License

MIT
