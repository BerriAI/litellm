import Image from '@theme/IdealImage';

# Athina


:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::


[Athina](https://athina.ai/) is an evaluation framework and production monitoring platform for your LLM-powered app. Athina is designed to enhance the performance and reliability of AI applications through real-time monitoring, granular analytics, and plug-and-play evaluations.

<Image img={require('../../img/athina_dashboard.png')} />

## Getting Started

Use Athina to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

## Using Callbacks

First, sign up to get an API_KEY on the [Athina dashboard](https://app.athina.ai).

Use just 1 line of code, to instantly log your responses **across all providers** with Athina:

```python
litellm.success_callback = ["athina"]
```

### Complete code

```python
from litellm import completion

## set env variables
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"
os.environ["OPENAI_API_KEY"]= ""

# set callback
litellm.success_callback = ["athina"]

#openai call
response = completion(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]
) 
```

## Additional information in metadata
You can send some additional information to Athina by using the `metadata` field in completion. This can be useful for sending metadata about the request, such as the customer_id, prompt_slug, or any other information you want to track.

```python
#openai call with additional metadata
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata={
    "environment": "staging",
    "prompt_slug": "my_prompt_slug/v1"
  }
)
```

Following are the allowed fields in metadata, their types, and their descriptions:

* `environment: Optional[str]` - Environment your app is running in (ex: production, staging, etc). This is useful for segmenting inference calls by environment.
* `prompt_slug: Optional[str]` - Identifier for the prompt used for inference. This is useful for segmenting inference calls by prompt.
* `customer_id: Optional[str]` - This is your customer ID. This is useful for segmenting inference calls by customer.
* `customer_user_id: Optional[str]` - This is the end user ID. This is useful for segmenting inference calls by the end user.
* `session_id: Optional[str]` - is the session or conversation ID. This is used for grouping different inferences into a conversation or chain. [Read more].(https://docs.athina.ai/logging/grouping_inferences)
* `external_reference_id: Optional[str]` - This is useful if you want to associate your own internal identifier with the inference logged to Athina.
* `context: Optional[Union[dict, str]]` - This is the context used as information for the prompt. For RAG applications, this is the "retrieved" data. You may log context as a string or as an object (dictionary).
* `expected_response: Optional[str]` - This is the reference response to compare against for evaluation purposes. This is useful for segmenting inference calls by expected response.
* `user_query: Optional[str]` - This is the user's query. For conversational applications, this is the user's last message.


## Using a self hosted deployment of Athina

If you are using a self hosted deployment of Athina, you will need to set the `ATHINA_BASE_URL` environment variable to point to your self hosted deployment.

```python
...
os.environ["ATHINA_BASE_URL"]= "http://localhost:9000"
...
```

## Support & Talk with Athina Team

- [Schedule Demo 👋](https://cal.com/shiv-athina/30min)
- [Website 💻](https://athina.ai/?utm_source=litellm&utm_medium=website)
- [Docs 📖](https://docs.athina.ai/?utm_source=litellm&utm_medium=website)
- [Demo Video 📺](https://www.loom.com/share/d9ef2c62e91b46769a39c42bb6669834?sid=711df413-0adb-4267-9708-5f29cef929e3)
- Our emails ✉️ shiv@athina.ai, akshat@athina.ai, vivek@athina.ai
