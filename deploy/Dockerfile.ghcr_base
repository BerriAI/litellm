# Use the provided base image
FROM ghcr.io/berriai/litellm:main-latest

# Set the working directory to /app
WORKDIR /app

# Copy the configuration file into the container at /app
COPY config.yaml .

# Make sure your entrypoint.sh is executable
RUN chmod +x entrypoint.sh

# Expose the necessary port
EXPOSE 4000/tcp

# Override the CMD instruction with your desired command and arguments
CMD ["--port", "4000", "--config", "config.yaml", "--detailed_debug", "--run_gunicorn"]
