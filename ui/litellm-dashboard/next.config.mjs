import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const nextConfig = {
  output: "export",
  experimental: {
    useTypeScriptCli: false,
  },
  compiler: {
    removeConsole: process.env.NODE_ENV === "production" ? { exclude: ["error", "warn"] } : false,
  },
  // Required with output: "export" — default image optimizer runs only in server mode.
  // See https://nextjs.org/docs/messages/export-image-api
  images: {
    unoptimized: true,
  },
  // The proxy mounts the static export under /ui. Baking that into basePath lets
  // next/link and router.push do client-side navigation instead of full reloads.
  // server_root_path deployments must rebuild with UI_BASE_PATH=<root>/ui
  // (see build_ui_custom_path.sh).
  basePath: process.env.UI_BASE_PATH || "/ui",
  assetPrefix: "/litellm-asset-prefix",
  trailingSlash: true,
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
};

export default nextConfig;
