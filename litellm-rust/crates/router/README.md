`litellm-router` holds the experimental routing API scaffold shared by SDK and gateway callers. `Router::new` accepts `RouterConfig` with owned `DeploymentConfig` values and retains multiple deployments per alias. Each input contains a public model name, a provider `Deployment`, and an optional deployment ID; entries without an ID get an ordinal ID for that configuration

```rust
use litellm_router::{Deployment, Router, config::{DeploymentConfig, RouterConfig}};

let router = Router::new(RouterConfig {
    deployments: vec![DeploymentConfig {
        deployment_id: Some("primary-1".into()),
        model_name: "primary".into(),
        deployment: Deployment {
            model: "provider/model".into(),
            ..Deployment::default()
        },
    }],
    ..RouterConfig::default()
})?;
```

`Router::snapshot`, `reconfigure`, `start`, and `close` define lifecycle operations. Reconfiguration validates before replacing the catalog, and calls retain their original configuration generation. Snapshots and selector inputs omit credentials

`call` separates per-request progress from the router catalog. `selection` exposes candidate values and a narrow selector trait. `retry` defines outcome and retry decisions, while `config` holds strategy, retry, fallback, timeout, and cooldown inputs. `Attempt` transfers ownership into the execution layer, including a returned stream, and is consumed by terminal reporting

Built-in selection, retries, fallback execution, health accounting, capacity permits, custom Python selectors, and provider execution remain unimplemented. `Call::next_attempt` and `Attempt::finish` return `Error::NotImplemented` rather than pretend those policies work

The compatibility constructor `FromIterator<(String, Deployment)>` preserves the gateway's existing exact lookup and last-entry-wins behavior through `get`. The new catalog and selection context retain every deployment in a model group

The router owns configuration and routing decisions. Core and host crates own provider execution and stream delivery. The gateway supplies per-attempt authorization through `AttemptAuthorizer`, keeping authentication and caller budgets outside this crate

`litellm-config` owns the application configuration schema. The gateway projects its model list into provider deployments for compatibility lookup. The Python bridge projects `_v2` model-list inputs into `DeploymentConfig` values. The routing engine has no dependency on the application schema or its YAML parser
