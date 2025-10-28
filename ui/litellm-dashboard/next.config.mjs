/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "",
  assetPrefix: "/litellm-asset-prefix", // If a server_root_path is set, this will be overridden by runtime injection
};

nextConfig.experimental = {
  missingSuspenseWithCSRBailout: false,
};

export default nextConfig;
