FROM ollama/ollama as ollama

RUN echo "auto installing llama2"

# auto install ollama/llama2
RUN ollama serve & sleep 2 && ollama pull llama2

RUN echo "installing litellm"

RUN apt-get update

# Install Python
RUN apt-get install -y python3 python3-pip ca-certificates

# Add AWS RDS CA bundle to system trust store
ADD https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem /usr/local/share/ca-certificates/aws-rds-global-bundle.crt
RUN update-ca-certificates

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt

RUN python3 -m pip install litellm
COPY start.sh /start.sh
ENTRYPOINT [ "/bin/bash", "/start.sh" ]
