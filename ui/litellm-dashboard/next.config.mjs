import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEV_PROXY_TARGET = process.env.LITELLM_BACKEND_URL || "http://localhost:4000";

const nextConfig = {
  output: "export",
  basePath: "",
  assetPrefix: "/litellm-asset-prefix",
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
  // Dev-only: proxy API calls to the litellm backend to avoid CORS issues
  async rewrites() {
    return [
      {
        source: "/v1/:path*",
        destination: `${DEV_PROXY_TARGET}/v1/:path*`,
      },
      {
        source: "/v2/:path*",
        destination: `${DEV_PROXY_TARGET}/v2/:path*`,
      },
      {
        source: "/key/:path*",
        destination: `${DEV_PROXY_TARGET}/key/:path*`,
      },
      {
        source: "/team/:path*",
        destination: `${DEV_PROXY_TARGET}/team/:path*`,
      },
      {
        source: "/user/:path*",
        destination: `${DEV_PROXY_TARGET}/user/:path*`,
      },
      {
        source: "/model/:path*",
        destination: `${DEV_PROXY_TARGET}/model/:path*`,
      },
      {
        source: "/health/:path*",
        destination: `${DEV_PROXY_TARGET}/health/:path*`,
      },
      {
        source: "/callbacks/:path*",
        destination: `${DEV_PROXY_TARGET}/callbacks/:path*`,
      },
      {
        source: "/config/:path*",
        destination: `${DEV_PROXY_TARGET}/config/:path*`,
      },
      {
        source: "/mcp-rest/:path*",
        destination: `${DEV_PROXY_TARGET}/mcp-rest/:path*`,
      },
      {
        source: "/sso/:path*",
        destination: `${DEV_PROXY_TARGET}/sso/:path*`,
      },
      {
        source: "/.well-known/:path*",
        destination: `${DEV_PROXY_TARGET}/.well-known/:path*`,
      },
      {
        source: "/litellm/.well-known/:path*",
        destination: `${DEV_PROXY_TARGET}/litellm/.well-known/:path*`,
      },
      {
        source: "/in_product_nudges",
        destination: `${DEV_PROXY_TARGET}/in_product_nudges`,
      },
    ];
  },
};

export default nextConfig;
