# Use the specific Node.js v20.11.0 image
FROM node:20.18.1-alpine3.20

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY ./litellm-js/spend-logs/package*.json ./

# Install dependencies
RUN npm install

# Install Prisma globally
RUN npm install -g prisma

# Copy the rest of the application code
COPY ./litellm-js/spend-logs .

# Generate Prisma client
RUN npx prisma generate

# Expose the port that the Node.js server will run on
EXPOSE 3000

# Command to run the Node.js app with npm run dev
CMD ["npm", "run", "dev"]
