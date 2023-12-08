# Deploying LiteLLM Proxy

## Quick Start Docker Image: Github Container Registry

### Pull the litellm ghcr docker image
See the latest available ghcr docker image here:
https://github.com/berriai/litellm/pkgs/container/litellm

```shell
docker pull ghcr.io/berriai/litellm:main-v1.10.1
```

### Run the Docker Image
```shell
docker run ghcr.io/berriai/litellm:main-v1.10.0
```

#### Run the Docker Image with LiteLLM CLI args

See all supported CLI args [here](https://docs.litellm.ai/docs/proxy/cli): 

Here's how you can run the docker image and pass your config to `litellm`
```shell
docker run ghcr.io/berriai/litellm:main-v1.10.0 --config your_config.yaml
```

Here's how you can run the docker image and start litellm on port 8002 with `num_workers=8`
```shell
docker run ghcr.io/berriai/litellm:main-v1.10.0 --port 8002 --num_workers 8
```
  
#### Run the Docker Image using docker compose

**Step 1**

- (Recommended) Use the example file `docker-compose.example.yml` given in the project root. e.g. https://github.com/BerriAI/litellm/blob/main/docker-compose.example.yml

- Rename the file `docker-compose.example.yml` to `docker-compose.yml`.

Here's an example `docker-compose.yml` file
```yaml
version: "3.9"
services:
  litellm:
    image: ghcr.io/berriai/litellm:main
    ports:
      - "8000:8000" # Map the container port to the host, change the host port if necessary
    volumes:
      - ./litellm-config.yaml:/app/config.yaml # Mount the local configuration file
    # You can change the port or number of workers as per your requirements or pass any new supported CLI augument. Make sure the port passed here matches with the container port defined above in `ports` value
    command: [ "--config", "/app/config.yaml", "--port", "8000", "--num_workers", "8" ]

# ...rest of your docker-compose config if any
```

**Step 2**

Create a `litellm-config.yaml` file with your LiteLLM config relative to your `docker-compose.yml` file.

Check the config doc [here](https://docs.litellm.ai/docs/proxy/configs)

**Step 3**

Run the command `docker-compose up` or `docker compose up` as per your docker installation.

> Use `-d` flag to run the container in detached mode (background) e.g. `docker compose up -d`


Your LiteLLM container should be running now on the defined port e.g. `8000`.


## Deploy on Render https://render.com/

<iframe width="840" height="500" src="https://www.loom.com/embed/805964b3c8384b41be180a61442389a3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


## LiteLLM Proxy Performance

### Throughput - 30% Increase
LiteLLM proxy + Load Balancer gives **30% increase** in throughput compared to Raw OpenAI API
<Image img={require('../img/throughput.png')} />

### Latency Added - 0.00325 seconds
LiteLLM proxy adds **0.00325 seconds** latency as compared to using the Raw OpenAI API
<Image img={require('../img/latency.png')} />
