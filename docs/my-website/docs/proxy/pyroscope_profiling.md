# Grafana Pyroscope CPU profiling

LiteLLM proxy can send continuous CPU profiles to [Grafana Pyroscope](https://grafana.com/docs/pyroscope/latest/) when enabled via environment variables. This is optional and off by default.

## Quick start

1. **Install the optional dependency** (required only when enabling Pyroscope):

   ```bash
   pip install pyroscope-io
   ```

   Or install the proxy extra:

   ```bash
   pip install "litellm[proxy]"
   ```

2. **Set environment variables** before starting the proxy:

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `LITELLM_ENABLE_PYROSCOPE` | Yes (to enable) | Set to `true` to enable Pyroscope profiling. |
   | `PYROSCOPE_APP_NAME` | Yes (when enabled) | Application name shown in the Pyroscope UI. |
   | `PYROSCOPE_SERVER_ADDRESS` | Yes (when enabled) | Pyroscope server URL (e.g. `http://localhost:4040`). |
   | `PYROSCOPE_SAMPLE_RATE` | No | Sample rate (integer). If unset, the pyroscope-io library default is used. |

3. **Start the proxy**; profiling will begin automatically when the proxy starts.

   ```bash
   export LITELLM_ENABLE_PYROSCOPE=true
   export PYROSCOPE_APP_NAME=litellm-proxy
   export PYROSCOPE_SERVER_ADDRESS=http://localhost:4040
   litellm --config config.yaml
   ```

4. **View profiles** in the Pyroscope (or Grafana) UI and select your `PYROSCOPE_APP_NAME`.

## Notes

- **Optional dependency**: `pyroscope-io` is an optional dependency. If it is not installed and `LITELLM_ENABLE_PYROSCOPE=true`, the proxy will log a warning and continue without profiling.
- **Platform support**: The `pyroscope-io` package uses a native extension and is not available on all platforms (e.g. Windows is excluded by the package).
- **Other settings**: See [Configuration settings](/proxy/config_settings) for all proxy environment variables.
