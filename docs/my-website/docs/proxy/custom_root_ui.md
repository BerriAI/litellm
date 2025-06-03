# UI - Custom Root Path 

💥 Use this when you want to serve LiteLLM on a custom base url path like `https://localhost:4000/api/v1` 

## Usage

### 1. Set `SERVER_ROOT_PATH` in your .env

👉 Set `SERVER_ROOT_PATH` in your .env and this will be set as your server root path

```
export SERVER_ROOT_PATH="/api/v1"
```

### 2. Run the Proxy

```shell
litellm proxy --config /path/to/config.yaml
```

After running the proxy you can access it on `http://0.0.0.0:4000/api/v1/` (since we set `SERVER_ROOT_PATH="/api/v1"`)


### 3. Reserve the `/litellm` path

LiteLLM uses the `/litellm` path to discover the custom root path. So you need to reserve this path in your proxy.

If you are running the UI, it will query the `/litellm/.well-known/litellm-ui-config` endpoint to get the UI configuration.

So you need to reserve the `/litellm` path in your proxy.

You can see the results with:

```bash
curl http://0.0.0.0:4000/litellm/.well-known/litellm-ui-config
```

Expected result:

```json

{
  "server_root_path": "/api/v1",
  ...
}

```


### 4. Verify Running on correct path

<Image img={require('../../img/custom_root_path.png')} />

**That's it**, that's all you need to run the proxy on a custom root path


## Demo

[Here's a demo video](https://drive.google.com/file/d/1zqAxI0lmzNp7IJH1dxlLuKqX2xi3F_R3/view?usp=sharing) of running the proxy on a custom root path