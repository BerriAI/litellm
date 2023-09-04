# Ollama 
LiteLLM supports all models from [Ollama](https://github.com/jmorganca/ollama)

### Ollama Models
Ollama supported models: https://github.com/jmorganca/ollama

| Model Name           | Function Call                                                                     | Required OS Variables          |
|----------------------|-----------------------------------------------------------------------------------|--------------------------------|
| Llama2 7B            | `completion(model='llama2', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 13B           | `completion(model='llama2:13b', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 70B           | `completion(model='llama2:70b', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 Uncensored    | `completion(model='llama2-uncensored', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Orca Mini            | `completion(model='orca-mini', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Vicuna               | `completion(model='vicuna', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Nous-Hermes          | `completion(model='nous-hermes', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Nous-Hermes 13B     | `completion(model='nous-hermes:13b', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Wizard Vicuna Uncensored | `completion(model='wizard-vicuna', messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
