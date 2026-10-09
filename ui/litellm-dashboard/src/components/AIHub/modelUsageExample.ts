import { isDecisionMode } from "@/lib/decisionModels";

const chatCompletionsExample = (baseUrl: string, model: string): string => `import openai

client = openai.OpenAI(
    api_key="your_api_key",
    base_url=${JSON.stringify(baseUrl)}  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model=${JSON.stringify(model)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`;

const systemOneExample = (baseUrl: string, model: string): string => `import requests

response = requests.post(
    ${JSON.stringify(`${baseUrl}/v1/systemone`)},  # Your LiteLLM Proxy URL
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(model)},
        "state": "I was charged twice for my subscription this month.",
        "questions": {
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?"
            }
        }
    }
)

print(response.json()["answers"])`;

export const modelUsageExample = (mode: string | null | undefined, baseUrl: string, model: string): string =>
  isDecisionMode(mode) ? systemOneExample(baseUrl, model) : chatCompletionsExample(baseUrl, model);
