# Use the provided base image
FROM ghcr.io/berriai/litellm-database:main-latest

# Set the working directory to /app
WORKDIR /app

# use this to allow prisma to run as non-root user
RUN mkdir -p /.cache
RUN chmod -R 777 /.cache

# Grant read access to all users for the site-packages directory
RUN chmod -R 777 /usr/local/lib/python3.11/site-packages
ENV PRISMA_BINARY_CACHE_DIR=/app/prisma

# Install Prisma CLI and generate Prisma client
# Use this to prevent installing prisma cli on the container startup
RUN pip install nodejs-bin 
RUN pip install prisma 

# Grant read access to all users for the site-packages directory
RUN chmod -R 777 /usr/local/lib/python3.11/site-packages


RUN prisma generate

# Expose the necessary port
EXPOSE 4000

# Define the command to run your app
ENTRYPOINT ["litellm"]
CMD ["--port", "4000"]
