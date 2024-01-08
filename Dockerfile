
# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.9-slim

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE as runtime
ARG with_database

WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=runtime /app/dist/*.whl .
COPY --from=runtime /wheels/ /wheels/

# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install --no-cache-dir --find-links=/wheels/ -r requirements.txt \
    && pip install *.whl \
    && rm -f *.whl


# Check if the with_database argument is set to 'true'
RUN echo "Value of with_database is: ${with_database}"
# If true, execute the following instructions
RUN if [ "$with_database" = "true" ]; then \
      prisma generate; \
      chmod +x /app/retry_push.sh; \
      /app/retry_push.sh; \
    fi

EXPOSE 8000/tcp

# Set your entrypoint and command
ENTRYPOINT ["litellm"]
CMD ["--config", "./proxy_server_config.yaml", "--port", "8000", "--num_workers", "8"]