# Lunary - Logging and tracing LLM input/output

:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::


[Lunary](https://lunary.ai/) is an open-source AI developer platform providing observability, prompt management, and evaluation tools for AI developers.

<video controls width='900' >
  <source src='https://lunary.ai/videos/demo-annotated.mp4'/>
</video>

## Use Lunary to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

:::info
We want to learn how we can make the callbacks better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

### Using Callbacks

First, sign up to get a public key on the [Lunary dashboard](https://lunary.ai).

Use just 2 lines of code, to instantly log your responses **across all providers** with lunary:

```python
litellm.success_callback = ["lunary"]
litellm.failure_callback = ["lunary"]
```

Complete code

```python
from litellm import completion

## set env variables
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key"

os.environ["OPENAI_API_KEY"] = ""

# set callbacks
litellm.success_callback = ["lunary"]
litellm.failure_callback = ["lunary"]

#openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
  user="ishaan_litellm"
)
```

## Templates

You can use Lunary to manage prompt templates and use them across all your LLM providers.

Make sure to have `lunary` installed:

```bash
pip install lunary
```

Then, use the following code to pull templates into Lunary:

```python
from litellm import completion
from lunary

template = lunary.render_template("template-slug", {
  "name": "John", # Inject variables
})

litellm.success_callback = ["lunary"]

result = completion(**template)
```

## Support & Talk to Founders

- Meet the Lunary team via [email](mailto:hello@lunary.ai).
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
