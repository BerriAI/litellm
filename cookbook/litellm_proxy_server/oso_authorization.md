# Oso authorization for model invocations

LiteLLM can ask Oso whether an authenticated caller may invoke a requested model

Authentication still belongs to LiteLLM. Virtual keys, JWT, SSO, OAuth2, and custom authentication establish the caller identity first. Oso only makes the authorization decision after that identity is available

The integration applies to authenticated `POST` requests that target a model, including Chat Completions, Responses, Anthropic Messages, embeddings, images, and built-in provider passthrough routes. It does not replace LiteLLM model allowlists or other built-in authorization checks

## Requirements

No additional Python package is required. LiteLLM calls Oso's Check API with its shared asynchronous HTTP client

Create an Oso API key and make it available to the proxy without putting the value in `config.yaml`

```shell
export OSO_AUTH="your-oso-api-key"
```

## Configuration

Add `oso_authorization` under `general_settings`

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  oso_authorization:
    enabled: true
    api_key: os.environ/OSO_AUTH
    url: https://api.osohq.com
    timeout: 5
```

The integration is disabled when the block is absent or `enabled` is `false`. `timeout` is in seconds, must be greater than 0, and cannot exceed 30

Requests to the main model invocation endpoints must include an explicit model while Oso authorization is enabled. This prevents a missing model from bypassing authorization through a proxy-side default

## Oso values

LiteLLM sends the following authorization request for each requested model

```text
actor: User:<user_id>, Team:<team_id>, or ApiKey:<one-way hash>
action: invoke
resource: Model:<requested model>
```

User identity takes precedence over team identity, and team identity takes precedence over API-key identity. When present, team, organization, project, and API-key identifiers are also sent as request-scoped `has_relation` context facts

The original LiteLLM API key is never included in the Oso request. The API-key identifier is a one-way hash when LiteLLM does not already hold a hashed key identity

## Example policy

This policy permits direct user or team grants and lets a user inherit a team's model grant from the context sent by LiteLLM

```polar
actor User {}
actor Team {}
actor ApiKey {}

resource Model {
  roles = ["invoker"];
  permissions = ["invoke"];

  "invoke" if "invoker";
}

has_permission(user: User, "invoke", model: Model) if
  has_relation(user, "team", team: Team) and
  has_role(team, "invoker", model);
```

An allow fact for a user can look like this

```polar
has_role(User{"alice"}, "invoker", Model{"gpt-5.4"});
```

With that fact, `User:alice` may invoke `gpt-5.4`. A request from the same user for another model is denied unless another policy rule or fact grants it

## Failure behavior

An Oso deny returns HTTP 403 through LiteLLM's existing proxy exception path

Invalid Oso configuration, timeout, connection failure, non-success response, or malformed response returns HTTP 503. These failures are fail-closed and never allow the model request to continue

An enabled integration also denies a model request when LiteLLM cannot construct a safe authenticated actor. Non-model routes and non-`POST` routes do not trigger an Oso model-invocation check

## Security notes

Keep `OSO_AUTH` in an environment variable or a LiteLLM-supported secret manager. Do not place it directly in source control

Oso runs in the central authentication dependency before budget reservation and before the provider request. Pre-call callbacks and `CustomLogger` hooks are not used as the enforcement point because some supported request paths can bypass those hooks

LiteLLM's existing model access, role, budget, route, and object permission checks remain active. Oso adds another required authorization decision and does not widen existing access
