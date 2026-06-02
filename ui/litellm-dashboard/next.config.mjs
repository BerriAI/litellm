import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// `output: "export"` is intentionally omitted here. Static export is incompatible
// with `dynamicParams: true` (the App Router default), which broke routes like
// /agents/[agent_id] and /agents/[agent_id]/sessions/[session_id]. The proxy
// production build is produced by `npm run build` separately when needed; in
// dev (`pnpm dev`) and standard server builds, dynamic routes work normally.
// See https://nextjs.org/docs/app/building-your-application/deploying/static-exports
const nextConfig = {
  images: {
    unoptimized: true,
  },
  basePath: "",
  assetPrefix: "/litellm-asset-prefix",
  trailingSlash: true,
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
};

export default nextConfig;
