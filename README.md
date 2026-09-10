# LIT-5777: dotted MCP arguments in Playground

Before: `b7dad8b44e30e87e6ae174ea6f58054e17cfc8a5`. After: `0b21b99ebd193872c520996cec79defad07ee108`

Both proxy images use the root Dockerfile, which compiles the dashboard from the corresponding source. This fixture is a real MCP SDK server using Streamable HTTP and echoing received arguments. No network mocks or LLM calls are involved

## Setup

1. Check out each revision into a separate directory, then run `docker build -t litellm-5777:before .` in the before directory and `docker build -t litellm-5777:after .` in the after directory
2. Put this directory's `compose.yaml`, `config.yaml`, and `server.py` together in a local directory. Create a private `.env` with random values for `LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`, and `POSTGRES_PASSWORD`
3. Run `REVISION=before docker compose up -d`. The gateway is bound to localhost port 45777, with its own PostgreSQL database and MCP server
4. Visit http://localhost:45777/ui/login/ and log in with username `admin` and your local master key as the password

## Matching Playground steps

1. Visit http://localhost:45777/ui/playground/
2. Set Endpoint Type to `/mcp-rest/tools/call`, MCP Server to `dotted_echo`, and Select Tool to `echo_required`
3. Enter `invoices` in `filter.category`, `September` in `query`, and `{"region":"eu"}` in the `options` JSON field
4. Click the send arrow. Before reports `Please enter filter.category` below the filled field and makes no tool call. After returns HTTP 200 and echoes `{"filter.category":"invoices","query":"September","options":{"region":"eu"}}`
5. Clear Chat and select `echo_optional`. Enter the same three values and click send. Before returns HTTP 200 but the arguments and echo contain `"filter":{"category":"invoices"}`. After returns HTTP 200 with the exact literal `"filter.category":"invoices"` key
6. Switch to `REVISION=after docker compose up -d gateway`, reload Playground, and repeat steps 1 through 5

The `*-network.json` files record the actual browser request payload, HTTP status, and MCP response. Screenshots show the real Playground without image editing. Browser control and screenshots use headless Chrome, so no desktop window is activated

## Checks

The existing form test file was extended and classified as an integration test because it renders the real form controls. The four added cases fail against the original implementation; all 18 cases pass with the fix. They cover exact submitted keys, prefix collisions with real objects, numeric conversion, required and JSON validation, the params wrapper, defaults, and switching tools

Dashboard formatting, changed-file ESLint, whole-dashboard lint budgets, and Knip are checked locally. The root Dockerfile runs the production Next.js build. Backend lint and type budget gates are skipped by the repository's change classification for this UI-only diff
