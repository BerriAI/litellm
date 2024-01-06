
# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.9-slim
# Builder stage
FROM $LITELLM_BUILD_IMAGE as builder

	@@ -35,8 +34,12 @@ RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE as runtime
ARG with_database

WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
COPY --from=builder /app/dist/*.whl .
	@@ -45,9 +48,17 @@ COPY --from=builder /wheels/ /wheels/
# Install the built wheel using pip; again using a wildcard if it's the only file
RUN pip install *.whl /wheels/* --no-index --find-links=/wheels/ && rm -f *.whl && rm -rf /wheels

# Check if the with_database argument is set to 'true'
RUN echo "Value of with_database is: ${with_database}"
# If true, execute the following instructions
RUN if [ "$with_database" = "true" ]; then \
      prisma generate; \
      chmod +x /app/retry_push.sh; \
      /app/retry_push.sh; \
    fi

EXPOSE 4000/tcp

# Set your entrypoint and command
ENTRYPOINT ["litellm"]
CMD ["--port", "4000"]