import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const devProxyUrl = process.env.LENS_DEV_PROXY_URL;

const nextConfig = {
  transpilePackages: ["@litellm/lens-ui"],
  ...(devProxyUrl
    ? {
        async rewrites() {
          return {
            beforeFiles: [
              // Every dashboard HTTP client sends Accept: application/json; page loads and RSC
              // fetches do not. That is what keeps GET /lens (API) apart from /lens (page) in dev.
              {
                source: "/:path*",
                has: [{ type: "header", key: "accept", value: "application/json.*" }],
                destination: `${devProxyUrl}/:path*`,
              },
              { source: "/ui/:path*", destination: "/:path*" },
            ],
            fallback: [{ source: "/:path*", destination: `${devProxyUrl}/:path*` }],
          };
        },
      }
    : {}),
  output: devProxyUrl ? undefined : "export",
  typescript: { tsconfigPath: "tsconfig.production.json" },
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
  basePath: "",
  assetPrefix: "/litellm-asset-prefix",
  trailingSlash: !devProxyUrl,
  skipTrailingSlashRedirect: Boolean(devProxyUrl),
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
};

export default nextConfig;
