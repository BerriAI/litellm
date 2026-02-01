/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "",
  assetPrefix: "/litellm-asset-prefix", // If a server_root_path is set, this will be overridden by runtime injection
  turbopack: {
    root: ".", // Explicitly set the project root to silence the multiple lockfiles warning
  },
};

export default nextConfig;
