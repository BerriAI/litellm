# Helm Chart for LiteLLM

> [!IMPORTANT]
> This is community maintained, Please make an issue if you run into a bug
> We recommend using [Docker or Kubernetes for production deployments](https://docs.litellm.ai/docs/proxy/prod)

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8.0+

If `db.deployStandalone` is used:
- PV provisioner support in the underlying infrastructure

If `db.useStackgresOperator` is used (not yet implemented):
- The Stackgres Operator must already be installed in the Kubernetes Cluster.  This chart will **not** install the operator if it is missing.

## Parameters

### LiteLLM Proxy Deployment Settings

| Name                                                       | Description                                                                                                                                                                           | Value |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `replicaCount`                                             | The number of LiteLLM Proxy pods to be deployed                                                                                                                                       | `1`  |
| `masterkey`                                                | The Master API Key for LiteLLM.  If not specified, a random key is generated.                                                                                                         | N/A  |
| `environmentSecrets`                                       | An optional array of Secret object names.  The keys and values in these secrets will be presented to the LiteLLM proxy pod as environment variables.  See below for an example Secret object.  | `[]`  |
| `environmentConfigMaps`                                       | An optional array of ConfigMap object names.  The keys and values in these configmaps will be presented to the LiteLLM proxy pod as environment variables.  See below for an example Secret object.  | `[]`  |
| `image.repository`                                         | LiteLLM Proxy image repository                                                                                                                                                        | `ghcr.io/berriai/litellm`  |
| `image.pullPolicy`                                         | LiteLLM Proxy image pull policy                                                                                                                                                       | `IfNotPresent`  |
| `image.tag`                                                | Overrides the image tag whose default the latest version of LiteLLM at the time this chart was published.                                                                             | `""`  |
| `imagePullSecrets`                                         | Registry credentials for the LiteLLM and initContainer images.                                                                                                                        | `[]`  |
| `serviceAccount.create`                                    | Whether or not to create a Kubernetes Service Account for this deployment.  The default is `false` because LiteLLM has no need to access the Kubernetes API.                          | `false`  |
| `service.type`                                             | Kubernetes Service type (e.g. `LoadBalancer`, `ClusterIP`, etc.)                                                                                                                      | `ClusterIP`  |
| `service.port`                                             | TCP port that the Kubernetes Service will listen on.  Also the TCP port within the Pod that the proxy will listen on.                                                                 | `4000`  |
| `ingress.*`                                                | See [values.yaml](./values.yaml) for example settings                                                                                                                                 | N/A  |
| `proxy_config.*`                                           | See [values.yaml](./values.yaml) for default settings.  See [example_config_yaml](../../../litellm/proxy/example_config_yaml/) for configuration examples.                            | N/A  |
| `extraContainers[]`                                        | An array of additional containers to be deployed as sidecars alongside the LiteLLM Proxy.                                                                                             | `[]`  |

#### Example `environmentSecrets` Secret 

```
apiVersion: v1
kind: Secret
metadata:
  name: litellm-envsecrets
data:
  AZURE_OPENAI_API_KEY: TXlTZWN1cmVLM3k=
type: Opaque
```

### Database Settings
| Name                                                       | Description                                                                                                                                                                           | Value |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `db.useExisting`                                           | Use an existing Postgres database.  A Kubernetes Secret object must exist that contains credentials for connecting to the database.  An example secret object definition is provided below.  | `false`  |
| `db.endpoint`                                              | If `db.useExisting` is `true`, this is the IP, Hostname or Service Name of the Postgres server to connect to.                                                                         | `localhost`  |
| `db.database`                                              | If `db.useExisting` is `true`, the name of the existing database to connect to.                                                                                                       | `litellm`  |
| `db.url`                                              | If `db.useExisting` is `true`, the connection url of the existing database to connect to can be overwritten with this value.                                                                                                       | `postgresql://$(DATABASE_USERNAME):$(DATABASE_PASSWORD)@$(DATABASE_HOST)/$(DATABASE_NAME)`  |
| `db.secret.name`                                           | If `db.useExisting` is `true`, the name of the Kubernetes Secret that contains credentials.                                                                                           | `postgres`  |
| `db.secret.usernameKey`                                    | If `db.useExisting` is `true`, the name of the key within the Kubernetes Secret that holds the username for authenticating with the Postgres instance.                                | `username`  |
| `db.secret.passwordKey`                                    | If `db.useExisting` is `true`, the name of the key within the Kubernetes Secret that holds the password associates with the above user.                                               | `password`  |
| `db.useStackgresOperator`                                  | Not yet implemented.                                                                                                                                                                  | `false`  |
| `db.deployStandalone`                                      | Deploy a standalone, single instance deployment of Postgres, using the Bitnami postgresql chart.  This is useful for getting started but doesn't provide HA or (by default) data backups.  | `true`  |
| `postgresql.*`                                             | If `db.deployStandalone` is `true`, configuration passed to the Bitnami postgresql chart.   See the [Bitnami Documentation](https://github.com/bitnami/charts/tree/main/bitnami/postgresql) for full configuration details.  See [values.yaml](./values.yaml) for the default configuration.  | See [values.yaml](./values.yaml) |
| `postgresql.auth.*`                                        | If `db.deployStandalone` is `true`, care should be taken to ensure the default `password` and `postgres-password` values are **NOT** used.                                            | `NoTaGrEaTpAsSwOrD`  |

#### Example Postgres `db.useExisting` Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres
data:
  # Password for the "postgres" user
  postgres-password: <some secure password, base64 encoded>
  username: litellm
  password: <some secure password, base64 encoded>
type: Opaque
```

#### Examples for `environmentSecrets` and `environemntConfigMaps`

```yaml
# Use config map for not-secret configuration data
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-env-configmap
data:
  SOME_KEY: someValue
  ANOTHER_KEY: anotherValue
```

```yaml
# Use secrets for things which are actually secret like API keys, credentials, etc
# Base64 encode the values stored in a Kubernetes Secret: $ pbpaste | base64 | pbcopy
# The --decode flag is convenient: $ pbpaste | base64 --decode

apiVersion: v1
kind: Secret
metadata:
  name: litellm-env-secret
type: Opaque
data:
  SOME_PASSWORD: cDZbUGVXeU5e0ZW    # base64 encoded
  ANOTHER_PASSWORD: AAZbUGVXeU5e0ZB # base64 encoded
```

Source: [GitHub Gist from troyharvey](https://gist.github.com/troyharvey/4506472732157221e04c6b15e3b3f094)

## Accessing the Admin UI
When browsing to the URL published per the settings in `ingress.*`, you will
be prompted for **Admin Configuration**.  The **Proxy Endpoint** is the internal
(from the `litellm` pod's perspective) URL published by the `<RELEASE>-litellm`
Kubernetes Service.  If the deployment uses the default settings for this
service, the **Proxy Endpoint** should be set to `http://<RELEASE>-litellm:4000`.

The **Proxy Key** is the value specified for `masterkey` or, if a `masterkey`
was not provided to the helm command line, the `masterkey` is a randomly
generated string stored in the `<RELEASE>-litellm-masterkey` Kubernetes Secret.

```bash
kubectl -n litellm get secret <RELEASE>-litellm-masterkey -o jsonpath="{.data.masterkey}"
```

## Admin UI Limitations
At the time of writing, the Admin UI is unable to add models.  This is because
it would need to update the `config.yaml` file which is a exposed ConfigMap, and
therefore, read-only.  This is a limitation of this helm chart, not the Admin UI
itself.
