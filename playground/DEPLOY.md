# Fly deployment

The Fly Machine runs the playground and OpenCode Web against the same workspace on one persistent volume

From the repository root, update the `app` name in `playground/fly.toml`, then create the app and volume

```sh
fly apps create litellm-rust-playground
fly volumes create playground_data --app litellm-rust-playground --region sjc --size 20
```

Set the OpenCode password and whichever provider key OpenCode should use

```sh
fly secrets set --app litellm-rust-playground \
  OPENCODE_SERVER_PASSWORD=replace-me \
  OPENAI_API_KEY=replace-me
```

Deploy with the repository root as the Docker build context

```sh
fly deploy --config playground/fly.toml .
```

The playground is served at `https://litellm-rust-playground.fly.dev` with the username `opencode`

OpenCode Web is served from the same Machine at `https://litellm-rust-playground.fly.dev:8443` with the username `opencode`

The first boot copies the LiteLLM Rust source, examples, and guide into `/data/workspace/litellm`. Later deployments leave that workspace unchanged

OpenCode session data is stored under `/data/opencode`. Provider keys and the OpenCode password stay in Fly secrets
