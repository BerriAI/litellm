<<<<<<< HEAD
# Proxy Server for Azure, Llama2, OpenAI, Claude, Hugging Face, Replicate Models
[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![PyPI Version](https://img.shields.io/badge/stable%20version-v0.1.345-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/)](https://pypi.org/project/litellm/0.1.1/)
![Downloads](https://img.shields.io/pypi/dm/litellm)
[![litellm](https://img.shields.io/badge/%20%F0%9F%9A%85%20liteLLM-OpenAI%7CAzure%7CAnthropic%7CPalm%7CCohere%7CReplicate%7CHugging%20Face-blue?color=green)](https://github.com/BerriAI/litellm)
=======
# Proxy Server for Chat API
>>>>>>> d1ff082 (new v litellm for render)

This repository contains a proxy server that interacts with OpenAI's Chat API and other similar APIs to facilitate chat-based language models. The server allows you to easily integrate chat completion capabilities into your applications. The server is built using Python and the Flask framework.

<<<<<<< HEAD
# Proxy Server for Chat API

This repository contains a proxy server that interacts with OpenAI's Chat API and other similar APIs to facilitate chat-based language models. The server allows you to easily integrate chat completion capabilities into your applications. The server is built using Python and the Flask framework.

## Installation

=======
## Installation

>>>>>>> d1ff082 (new v litellm for render)
To set up and run the proxy server locally, follow these steps:

1. Clone this repository to your local machine:


2. Install the required dependencies using pip:

`pip install -r requirements.txt`

3. Configure the server settings, such as API keys and model endpoints, in the configuration file (`config.py`).

4. Run the server:

`python app.py`


## API Endpoints

### `/chat/completions` (POST)

This endpoint is used to generate chat completions. It takes in JSON data with the following parameters:

- `model` (string, required): ID of the model to use for chat completions. Refer to the model endpoint compatibility table for supported models.
- `messages` (array, required): A list of messages representing the conversation context. Each message should have a `role` (system, user, assistant, or function), `content` (message text), and `name` (for function role).
- Additional parameters for controlling completions, such as `temperature`, `top_p`, `n`, etc.

Example JSON payload:

```json
{
"model": "gpt-3.5-turbo",
"messages": [
 {"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "Knock knock."},
 {"role": "assistant", "content": "Who's there?"},
 {"role": "user", "content": "Orange."}
],
"temperature": 0.8
}
```


## Input Parameters
model: ID of the language model to use.
messages: An array of messages representing the conversation context.
role: The role of the message author (system, user, assistant, or function).
content: The content of the message.
name: The name of the author (required for function role).
function_call: The name and arguments of a function to call.
functions: A list of functions the model may generate JSON inputs for.
Various other parameters for controlling completion behavior.
Supported Models
The proxy server supports the following models:

OpenAI Chat Completion Models:
gpt-4
gpt-4-0613
gpt-4-32k
...
OpenAI Text Completion Models:
text-davinci-003
Cohere Models:
command-nightly
command
...
Anthropic Models:
claude-2
claude-instant-1
...
Replicate Models:
replicate/
OpenRouter Models:
google/palm-2-codechat-bison
google/palm-2-chat-bison
...
Vertex Models:
chat-bison
chat-bison@001
<<<<<<< HEAD
Refer to the model endpoint compatibility table for more details.
=======
Refer to the model endpoint compatibility table for more details.
>>>>>>> d1ff082 (new v litellm for render)
