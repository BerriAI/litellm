import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Predicted Outputs

| Property | Details |
|-------|-------|
| Description | Use this when most of the output of the LLM is known ahead of time. For instance, if you are asking the model to rewrite some text or code with only minor changes, you can reduce your latency significantly by using Predicted Outputs, passing in the existing content as your prediction. |
| Supported providers | `openai` |
| Link to OpenAI doc on Predicted Outputs | [Predicted Outputs ↗](https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outputs) |
| Supported from LiteLLM Version | `v1.51.4` |



## Using Predicted Outputs

<Tabs>
<TabItem label="LiteLLM Python SDK" value="Python">

In this example we want to refactor a piece of C# code, and convert the Username property to Email instead:
```python
import litellm
os.environ["OPENAI_API_KEY"] = "your-api-key"
code = """
/// <summary>
/// Represents a user with a first name, last name, and username.
/// </summary>
public class User
{
    /// <summary>
    /// Gets or sets the user's first name.
    /// </summary>
    public string FirstName { get; set; }

    /// <summary>
    /// Gets or sets the user's last name.
    /// </summary>
    public string LastName { get; set; }

    /// <summary>
    /// Gets or sets the user's username.
    /// </summary>
    public string Username { get; set; }
}
"""

completion = litellm.completion(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
        },
        {"role": "user", "content": code},
    ],
    prediction={"type": "content", "content": code},
)

print(completion)
```

</TabItem>
<TabItem label="LiteLLM Proxy Server" value="proxy">

1. Define models on config.yaml

```yaml
model_list:
  - model_name: gpt-4o-mini # OpenAI gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY 

```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK


```python
from openai import OpenAI

client = OpenAI(
    api_key="LITELLM_PROXY_KEY", # sk-1234
    base_url="LITELLM_PROXY_BASE" # http://0.0.0.0:4000
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
        },
        {"role": "user", "content": code},
    ],
    prediction={"type": "content", "content": code},
)

print(completion)
```

</TabItem>
</Tabs>
