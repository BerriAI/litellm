# Using GPTCache with LiteLLM

GPTCache is a Library for Creating Semantic Cache for LLM Queries

GPTCache Docs: https://gptcache.readthedocs.io/en/latest/index.html#
GPTCache Github: https://github.com/zilliztech/GPTCache

## Usage

### Install GPTCache
pip install gptcache

### Using GPT Cache with Litellm Completion()

```python
from gptcache import cache
from litellm.cache import completion

# Set your .env keys 
os.environ['OPENAI_API_KEY'] = ""
cache.init()
cache.set_openai_key()

messages = [{"role": "user", "content": "what is litellm YC 22?"}]
```

