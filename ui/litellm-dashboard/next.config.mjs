/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',
    basePath: process.env.UI_BASE_PATH || '/ui',
};

export default nextConfig;
