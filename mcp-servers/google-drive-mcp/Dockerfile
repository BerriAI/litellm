# Use Node.js LTS as base image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only, skipping lifecycle scripts
RUN npm ci --only=production --ignore-scripts

# Copy built distribution files
COPY dist ./dist

# Create directory for config files
RUN mkdir -p /config

# Set environment variables
ENV NODE_ENV=production
ENV GOOGLE_DRIVE_OAUTH_CREDENTIALS=/config/gcp-oauth.keys.json
ENV GOOGLE_DRIVE_MCP_TOKEN_PATH=/config/tokens.json

# Make the main script executable
RUN chmod +x dist/index.js

# Run as non-root user
USER node

# Start the server
ENTRYPOINT ["node", "dist/index.js"]