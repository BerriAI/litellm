/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',
    basePath: '',
    assetPrefix: '/litellm',  // This will make assets load from /litellm/_next/...
};

nextConfig.experimental = {
    missingSuspenseWithCSRBailout: false
}

export default nextConfig;
