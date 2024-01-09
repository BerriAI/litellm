import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🐳 Docker, Deploying LiteLLM Proxy

You can find the Dockerfile to build litellm proxy [here](https://github.com/BerriAI/litellm/blob/main/Dockerfile)

## Quick Start

See the latest available ghcr docker image here:
https://github.com/berriai/litellm/pkgs/container/litellm

```shell
docker pull ghcr.io/berriai/litellm:main-latest
```

```shell
docker run ghcr.io/berriai/litellm:main-latest
```

### Run with LiteLLM CLI args

See all supported CLI args [here](https://docs.litellm.ai/docs/proxy/cli): 

Here's how you can run the docker image and pass your config to `litellm`
```shell
docker run ghcr.io/berriai/litellm:main-latest --config your_config.yaml
```

Here's how you can run the docker image and start litellm on port 8002 with `num_workers=8`
```shell
docker run ghcr.io/berriai/litellm:main-latest --port 8002 --num_workers 8
```

## Deploy with Database

We maintain a [seperate Dockerfile](https://github.com/BerriAI/litellm/pkgs/container/litellm-database) for reducing build time when running LiteLLM proxy with a connected Postgres Database 

<Tabs>
<TabItem value="docker-deploy" label="Dockerfile">

```
docker pull docker pull ghcr.io/berriai/litellm-database:main-v1.16.20
```

```
docker run --name litellm-proxy \
-e DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname> \
-p 4000:4000 \
ghcr.io/berriai/litellm-database:main-v1.16.20
```

Your OpenAI proxy server is now running on `http://0.0.0.0:4000`.

</TabItem>
<TabItem value="kubernetes-deploy" label="Kubernetes">

### Step 1. Create deployment.yaml

```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: litellm-deployment
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: litellm
     template:
       metadata:
         labels:
           app: litellm
       spec:
         containers:
           - name: litellm-container
             image: ghcr.io/berriai/litellm-database:main-v1.16.20
             env:
              - name: DATABASE_URL
                value: postgresql://<user>:<password>@<host>:<port>/<dbname>
```

```bash
kubectl apply -f /path/to/deployment.yaml
```

### Step 2. Create service.yaml 

```yaml
apiVersion: v1
kind: Service
metadata:
  name: litellm-service
spec:
  selector:
    app: litellm
  ports:
    - protocol: TCP
      port: 4000
      targetPort: 4000
  type: NodePort
```

```bash
kubectl apply -f /path/to/service.yaml
```

### Step 3. Start server

```
kubectl port-forward service/litellm-service 4000:4000
```

Your OpenAI proxy server is now running on `http://0.0.0.0:4000`.

</TabItem>
</Tabs>

## Platform-specific Guide


<Tabs>
<TabItem value="google-cloud-run" label="Google Cloud Run">

### Deploy on Google Cloud Run
**Click the button** to deploy to Google Cloud Run

[![Deploy](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run/?git_repo=https://github.com/BerriAI/litellm)

#### Testing your deployed proxy
**Assuming the required keys are set as Environment Variables**

https://litellm-7yjrj3ha2q-uc.a.run.app is our example proxy, substitute it with your deployed cloud run app

```shell
curl https://litellm-7yjrj3ha2q-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```


</TabItem>
<TabItem value="render" label="Render deploy">

### Deploy on Render https://render.com/

<iframe width="840" height="500" src="https://www.loom.com/embed/805964b3c8384b41be180a61442389a3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>



</TabItem>
<TabItem value="railway" label="Railway">

### Deploy on Railway https://railway.app

**Step 1: Click the button** to deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/S7P9sn?referralCode=t3ukrU)

**Step 2:** Set `PORT` = 4000 on Railway Environment Variables

</TabItem>
</Tabs>


## Extras 

### Run with docker compose

**Step 1**

- (Recommended) Use the example file `docker-compose.example.yml` given in the project root. e.g. https://github.com/BerriAI/litellm/blob/main/docker-compose.example.yml

- Rename the file `docker-compose.example.yml` to `docker-compose.yml`.

Here's an example `docker-compose.yml` file
```yaml
version: "3.9"
services:
  litellm:
    build:
      context: .
        args:
          target: runtime
    image: ghcr.io/berriai/litellm:main-latest
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



## LiteLLM Proxy Performance

LiteLLM proxy has been load tested to handle 1500 req/s.

### Throughput - 30% Increase
LiteLLM proxy + Load Balancer gives **30% increase** in throughput compared to Raw OpenAI API
<Image img={require('../../img/throughput.png')} />

### Latency Added - 0.00325 seconds
LiteLLM proxy adds **0.00325 seconds** latency as compared to using the Raw OpenAI API
<Image img={require('../../img/latency.png')} />
