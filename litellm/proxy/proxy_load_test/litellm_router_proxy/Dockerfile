# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app will run on
EXPOSE 8090

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]