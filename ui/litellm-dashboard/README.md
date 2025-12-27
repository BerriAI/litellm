## LiteLLM Dashboard

This is the Next.js app for the LiteLLM admin dashboard. Most builds ship the precompiled static output already committed to `proxy/_experimental/out`. Rebuild the UI only when you change dashboard code.

### Local dev
```bash
npm ci # or pnpm install --frozen-lockfile
npm run dev # or pnpm dev
```
Open http://localhost:3000.

### Rebuild the static UI (what Docker images use)
From this folder:
```bash
./build_ui.sh
```
This will:
- Use Node 20 (via `nvm use v20`)
- `npm run build`
- Copy the build output into `../..//litellm/proxy/_experimental/out`

### When to commit the build artifacts
If you change dashboard code and want downstream users (or Docker builds that skip UI rebuilds) to pick it up, run `./build_ui.sh` and commit the updated files in `litellm/proxy/_experimental/out`.

If you do not commit the build output, make sure your image pipeline runs the UI build step (e.g., `build_ui.sh` in Docker) so the new UI is included.
