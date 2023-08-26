# Create your first playground
import Image from '@theme/IdealImage';

Learn how to build a light version of the demo playground as shown on the website in less than 10 minutes. 

**What we'll build**: We'll build <u>the server</u> and connect it to our template frontend, ending up with a deployed playground by the end!


:::info

 Before you start with this section, make sure you have followed the [environment-setup](./installation) guide. Please note, that this demo relies on you having API keys from at least 1 model provider (E.g. OpenAI). 
:::

## 1. Test keys 

Let's make sure our keys are working. Run this script in any environment of your choice (e.g. [Google Colab](https://colab.research.google.com/#create=true)).

🚨 Don't forget to replace the placeholder key values with your keys!

```python 
pip install litellm
```

```python
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key" ## REPLACE THIS
os.environ["COHERE_API_KEY"] = "cohere key" ## REPLACE THIS
os.environ["AI21_API_KEY"] = "ai21 key" ## REPLACE THIS


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)

# ai21 call
response = completion("j2-mid", messages)
```

## 2. Set-up Server

### 2.1 Spin-up Template
Let's build a basic Flask app as our backend server.  

Create a `main.py` file, and put in this starter code.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# Example route
@app.route('/', methods=['GET'])
def hello():
    return jsonify(message="Hello, Flask!")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=4000, threads=500)
```

Let's test that it's working. 

Start the server:
```python 
python main.py
```

Run a curl command to test it:
```curl
curl -X GET localhost:4000
```

This is what you should see

<Image img={require('../../img/test_python_server_1.png')} alt="python_code_sample_1" />

### 2.2 Add `completion` route

Now, let's add a route for our completion calls. This is when we'll add litellm to our server to handle the model requests. 

**Notes**:
* 🚨 Don't forget to replace the placeholder key values with your keys!
* `completion_with_retries`: LLM API calls can fail in production. This function wraps the normal litellm completion() call with [tenacity](https://tenacity.readthedocs.io/en/latest/) to retry the call in case it fails. 

The snippet we'll add:

```python 
import os
from litellm import completion_with_retries 

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key" ## REPLACE THIS
os.environ["COHERE_API_KEY"] = "cohere key" ## REPLACE THIS
os.environ["AI21_API_KEY"] = "ai21 key" ## REPLACE THIS


@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    data["max_tokens"] = 256 # By default let's set max_tokens to 256
    try:
        # COMPLETION CALL
        response = completion_with_retries(**data)
    except Exception as e:
        # print the error
        print(e)
    return response
```

The complete code:

```python 
import os
from flask import Flask, jsonify, request
from litellm import completion_with_retries 


## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key" ## REPLACE THIS
os.environ["COHERE_API_KEY"] = "cohere key" ## REPLACE THIS
os.environ["AI21_API_KEY"] = "ai21 key" ## REPLACE THIS

app = Flask(__name__)

# Example route
@app.route('/', methods=['GET'])
def hello():
    return jsonify(message="Hello, Flask!")

@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    data["max_tokens"] = 256 # By default let's set max_tokens to 256
    try:
        # COMPLETION CALL
        response = completion_with_retries(**data)
    except Exception as e:
        # print the error
        print(e)

    return response

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=4000, threads=500)
```

Start the server:
```python 
python main.py
```

Run this curl command to test it:
```curl
curl -X POST localhost:4000/chat/completions \
-H 'Content-Type: application/json' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [{
    "content": "Hello, how are you?",
    "role": "user"
  }]
}'
```

This is what you should see

<Image img={require('../../img/test_python_server_2.png')} alt="python_code_sample_2" />

## 3. Connect to our frontend template


## 4. Deploy!


