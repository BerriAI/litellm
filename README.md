# LIT-7449 verification

Before and after source commits are in before-sha and after-sha. Live JSON files contain actual HTTP observations with credentials omitted. PNG results pages are explicitly labeled renderings of those records; consent PNGs are actual LiteLLM pages captured in headless Chrome

The isolated local Compose stack uses PostgreSQL and a source image for each revision. Both images reuse the same existing dependency/runtime image, with LiteLLM, enterprise, proxy extras and schema files copied from clean commit archives. Dependency changes since that base are the proxy-extras workspace version; its source was copied as well. This is a source-overlay build, not a fresh full dependency rebuild

To reproduce, build the gateway at each listed commit, configure the public Microsoft Learn MCP using config.yaml, set private LITELLM_MASTER_KEY, LITELLM_SALT_KEY, UI_USERNAME=admin and UI_PASSWORD equal to the local master key, and supply a local PostgreSQL DATABASE_URL. Expose only localhost:47449

Run bash repro.sh for the exact registration matrix. live_verify.py uses the same HTTP flow plus local sign-in and a headless browser to verify cookie storage and capture the real consent page. It requires Playwright and Chrome, a private .env, and the before-sha/after-sha files. Run python live_verify.py before or python live_verify.py after. It checks PKCE exchange, token refresh, replay rejection, initialization, three public Microsoft Learn tools and a read-only documentation search. No LLM invocation is required

The maximum-length case uses four distinct 256-character ASCII HTTPS URIs with normal state. It is not a claim that every extreme state/identity combination fits every browser or proxy. Larger encoded metadata remains rejected by the unchanged client-ID guard. The separate internal MCP v2 RFC 003 records comprehensive encoded-size policy and CIMD as future work

Local gateway tests: 142 passed. Related discoverable-endpoint tests: 373 passed. Changed executable-line coverage is 100%; register_aggregate_client branch coverage is 100%. Full module and repository coverage are separate; repository-wide coverage requires the completed hosted Codecov report

## Desktop acceptance

VS Code 1.136.2, commit88e44fa0e00b08f7758b4f6d05632e4fd5e4df6f, runs in an isolated Linux/Xvfb/Openbox desktop with a session bus. The browser and loopback callback run in the same container. The desktop connects to http://127.0.0.1:4000/mcp, forwarded to the matching gateway image. This avoids changing the host desktop

Before: open MCP: List Servers, select lit-7449, and Start Server. Registration returns400 and the native Dynamic Client Registration not supported dialog appears. After: repeat the same steps, allow authentication, sign in to LiteLLM, and click Finish connecting. VS Code receives the callback and discovers3tools

The small local verification extension calls the discovered Microsoft Learn documentation-search tool through vscode.lm.invokeTool and displays its actual response in VS Code Output. It does not emulate the MCP transport or call an LLM. Both the initial and post-refresh calls returned documentation

Refresh fixture: stop VS Code, move only created_at in its encrypted local token cache into the past with expire-desktop-cache.py, then reopen and start the server. The script operates on a private copy of /profile/User/globalStorage/state.vscdb named .private-vscode-session.vscdb. Copy the result back while VS Code is stopped. It requires the isolated test profile using --password-store=basic. Real access/refresh credentials and gateway source/settings remain unchanged. VS Code logs that it is refreshing, POST /token returns200, and the next real tool call succeeds without another login. This exercises cached-expiry handling, not an hour-long natural-expiration wait

All desktop screenshots are actual captures. The native baseline dialog and tool-call captures include the virtual display; consent and refresh captures show the actual application content. No video was recorded. Test containers, network and disposable database were removed afterward
