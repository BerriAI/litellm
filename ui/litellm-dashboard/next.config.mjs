/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',
    basePath: process.env.UI_BASE_PATH || '/ui',
    baseProxyUrl: process.env.PROXY_BASE_URL || '/',
};

nextConfig.experimental = {
    missingSuspenseWithCSRBailout: false
}

export default nextConfig;
