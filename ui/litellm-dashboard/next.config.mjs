import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const nextConfig = {
  output: "export",
  // Pin the directory-index layout (<route>/index.html). Otherwise different
  // Next.js patch versions or stale .next caches flip the export between
  // foo.html and foo/index.html, producing massive rename-only diffs in
  // litellm/proxy/_experimental/out and breaking deployments that rely on
  // directory-index routing for nested paths (e.g. /ui/mcp/oauth/callback).
  trailingSlash: true,
  // Required with output: "export" — default image optimizer runs only in server mode.
  // See https://nextjs.org/docs/messages/export-image-api
  images: {
    unoptimized: true,
  },
  basePath: "",
  assetPrefix: "/litellm-asset-prefix",
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
};

export default nextConfig;
