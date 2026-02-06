FROM python:3.14.0a3-slim

# Add AWS RDS CA bundle to system trust store
ADD https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem /usr/local/share/ca-certificates/aws-rds-global-bundle.crt
RUN update-ca-certificates

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE $PORT 

CMD litellm --host 0.0.0.0 --port $PORT --workers 10 --config config.yaml