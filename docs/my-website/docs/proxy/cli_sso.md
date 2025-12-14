# CLI Authentication

Use the litellm cli to authenticate to the LiteLLM Gateway. This is great if you're trying to give a large number of developers self-serve access to the LiteLLM Gateway.


## Demo

<iframe width="840" height="500" src="https://www.loom.com/embed/87c5d243cde642ff942783024ff037e3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Usage 

### Prerequisites - Start LiteLLM Proxy with Beta Flag

:::warning[Beta Feature - Required]

CLI SSO Authentication is currently in beta. You must set this environment variable **when starting up your LiteLLM Proxy**:

```bash
export EXPERIMENTAL_UI_LOGIN="True"
litellm --config config.yaml
```

Or add it to your proxy startup command:

```bash
EXPERIMENTAL_UI_LOGIN="True" litellm --config config.yaml
```

:::

### Steps

1. **Install the CLI**

   If you have [uv](https://github.com/astral-sh/uv) installed, you can try this:

   ```shell
   uv tool install 'litellm[proxy]'
   ```

   If that works, you'll see something like this:

   ```shell
   ...
   Installed 2 executables: litellm, litellm-proxy
   ```

   and now you can use the tool by just typing `litellm-proxy` in your terminal:

   ```shell
   litellm-proxy
   ```

2. **Set up environment variables**

   On your local machine, set the proxy URL:

   ```bash
   export LITELLM_PROXY_URL=http://localhost:4000
   ```

   *(Replace with your actual proxy URL)*

3. **Login**

   ```shell
   litellm-proxy login
   ```

   This will open a browser window to authenticate. If you have connected LiteLLM Proxy to your SSO provider, you should be able to login with your SSO credentials. Once logged in, you can use the CLI to make requests to the LiteLLM Gateway.

4. **Make a test request to view models**

   ```shell
   litellm-proxy models list
   ```

   This will list all the models available to you.