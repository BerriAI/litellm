# Compose deployments

The LiteAdmin overlay requires an explicit project name. Moving the Compose files changes the default project name and therefore the state volume name. Reuse the existing name so the agent keeps its saved state.

Set `EXISTING_LITEADMIN_CONTAINER` to the running or stopped agent container, then read its project label before upgrading.

```sh
export COMPOSE_PROJECT_NAME="$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project" }}' "$EXISTING_LITEADMIN_CONTAINER")"
docker compose -f litellm/deploy/compose/docker-compose.yml \
  -f litellm/deploy/compose/docker-compose.liteadmin.yml up -d
```

Passing the same existing name with `docker compose --project-name` also works. For a new deployment, choose a stable project name and keep using it.
