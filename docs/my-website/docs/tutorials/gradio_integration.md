# Gradio Chatbot + LiteLLM Tutorial
Simple tutorial for integrating LiteLLM completion calls with streaming Gradio chatbot demos

### Install & Import Dependencies
```python
!pip install gradio litellm
import gradio
import litellm
```

### Define Inference Function
Remember to set `model` and `api_base` as expected by the server hosting your LLM.
```python
def inference(message, history):
    try:
        flattened_history = [item for sublist in history for item in sublist]
        full_message = " ".join(flattened_history + [message])
        messages_litellm = [{"role": "user", "content": full_message}] # litellm message format
        partial_message = ""
        for chunk in litellm.completion(model="huggingface/meta-llama/Llama-2-7b-chat-hf",
                                        api_base="x.x.x.x:xxxx",
                                        messages=messages_litellm,
                                        max_new_tokens=512,
                                        temperature=.7,
                                        top_k=100,
                                        top_p=.9,
                                        repetition_penalty=1.18,
                                        stream=True):
            partial_message += chunk['choices'][0]['delta']['content'] # extract text from streamed litellm chunks
            yield partial_message
    except Exception as e:
        print("Exception encountered:", str(e))
        yield f"An Error occured please 'Clear' the error and try your question again"
```

### Define Chat Interface
```python
gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Enter text here...", container=False, scale=5),
    description=f"""
    CURRENT PROMPT TEMPLATE: {model_name}.
    An incorrect prompt template will cause performance to suffer.
    Check the API specifications to ensure this format matches the target LLM.""",
    title="Simple Chatbot Test Application",
    examples=["Define 'deep learning' in once sentence."],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    theme=theme,
).queue().launch()
```
### Launch Gradio App
1. From command line: `python app.py` or `gradio app.py` (latter enables live deployment updates)
2. Visit provided hyperlink in your browser.
3. Enjoy prompt-agnostic interaction with remote LLM server.

### Recommended Extensions:
* Add command line arguments to define target model & inference endpoints

Credits to [ZQ](https://x.com/ZQ_Dev), for this tutorial.