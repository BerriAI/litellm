# Output Format - completion()
Here's the exact json output and type you can expect from all litellm `completion` calls for all models

```python 
{
  'choices': [
    {
      'finish_reason': str,       # String: 'stop'
      'index': int,               # Integer: 0
      'message': {                # Dictionary [str, str]
        'role': str,              # String: 'assistant'
        'content': str            # String: "default message"
      }
    }
  ],
  'created': str,               # String: None
  'model': str,                 # String: None
  'usage': {                    # Dictionary [str, int]
    'prompt_tokens': int,       # Integer
    'completion_tokens': int,   # Integer
    'total_tokens': int         # Integer
  }
}

```