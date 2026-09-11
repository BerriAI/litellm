# LIT-7449 verification

Before and after source commits are in before-sha and after-sha. Live JSON files contain actual HTTP observations with credentials omitted. PNG results pages are explicitly labeled renderings of those records; consent PNGs are actual LiteLLM pages captured in headless Chrome

The isolated local Compose stack uses PostgreSQL and a source image for each revision. Both images reuse the same existing dependency/runtime image, with LiteLLM, enterprise, proxy extras and schema files copied from clean commit archives. Dependency changes since that base are the proxy-extras workspace version; its source was copied as well. This is a source-overlay build, not a fresh full dependency rebuild

To reproduce, build the gateway at each listed commit, configure the public Microsoft Learn MCP using config.yaml, set private LITELLM_MASTER_KEY, LITELLM_SALT_KEY, UI_USERNAME=admin and UI_PASSWORD equal to the local master key, and supply a local PostgreSQL DATABASE_URL. Expose only localhost:47449

Run bash repro.sh for the exact registration matrix. live_verify.py uses the same HTTP flow plus local sign-in and a headless browser to verify cookie storage and capture the real consent page. It requires Playwright and Chrome, a private .env, and the before-sha/after-sha files. Run python live_verify.py before or python live_verify.py after. It checks PKCE exchange, token refresh, replay rejection, initialization, three public Microsoft Learn tools and a read-only documentation search. No LLM invocation is required

The maximum-length case uses four distinct 256-character ASCII HTTPS URIs with normal state. It is not a claim that every extreme state/identity combination fits every browser or proxy. Larger encoded metadata remains rejected by the unchanged client-ID guard. The separate internal MCP v2 RFC 003 records comprehensive encoded-size policy and CIMD as future work

Local gateway tests: 142 passed. Related discoverable-endpoint tests: 373 passed. Changed executable-line coverage is 100%; register_aggregate_client branch coverage is 100%. Full module and repository coverage are separate; repository-wide coverage requires the completed hosted Codecov report
