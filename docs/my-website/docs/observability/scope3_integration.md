# Scope3 - Track Environmental Impact (CO2 and H2O)


:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::


[Scope3](https://ai.scope3.com/) provides an enterprise view of the environmental impact of digital supply chain, helping companies and AI developers understand how their choice of AI vendors impacts their carbon footprint and water intensity.

## Getting Started

1. [Generate an access token](https://aidocs.scope3.com/docs/authentication) and set it as an environment variable:

```
os.environ['SCOPE3_ACCESS_TOKEN'] = 'scope3_...'
```

2. Configure Scope3 as a success callback in your `litellm_config.yaml`:

```python
litellm.success_callback = ["scope3"]
```

3. (Optional) add metadata to associate the request with an environment, application, client, project, or session:

```python
  metadata={
    "scope3_environment": "production",
    "scope3_application_id": "creative_generation",
    "scope3_client_id": "diageo",
    "scope3_project_id": "q3_uk_campaign",
    "scope3_session_id": "366618fc-50df-47ce-a677-19da9a6dde8d"
  }
```

### Complete code

```python
from litellm import completion

## set env variables
os.environ["SCOPE3_ACCESS_TOKEN"] = "scope3_""
os.environ["OPENAI_API_KEY"]= ""

# set callback
litellm.success_callback = ["scope3"]

#openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
  metadata={
    "scope3_environment": "production",
    "scope3_application_id": "creative_generation",
    "scope3_client_id": "diageo",
    "scope3_project_id": "q3_prototype",
    "scope3_session_id": "366618fc-50df-47ce-a677-19da9a6dde8d"
  }
)
```

## Learn More & Talk with Scope3 Team

- [Website 💻](https://ai.scope3.com/?utm_source=litellm&utm_medium=website)
- [Docs 📖](https://aidocs.scope3.com/?utm_source=litellm&utm_medium=website)