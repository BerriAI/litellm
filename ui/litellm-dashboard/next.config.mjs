/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',
    basePath: '/ui',
};

nextConfig.experimental = {
    missingSuspenseWithCSRBailout: false
}

export default nextConfig;
