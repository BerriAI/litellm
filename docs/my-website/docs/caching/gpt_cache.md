# Using GPTCache with LiteLLM

GPTCache is a Library for Creating Semantic Cache for LLM Queries

GPTCache Docs: https://gptcache.readthedocs.io/en/latest/index.html#

GPTCache Github: https://github.com/zilliztech/GPTCache

## Usage

### Install GPTCache
```
pip install gptcache
```

### Using GPT Cache with Litellm Completion()

#### Using GPTCache
In order to use GPTCache the following lines are used to instantiat it
```python
from gptcache import cache
# set API keys in .env / os.environ
cache.init()
cache.set_openai_key()
```

#### Full Code using GPTCache and LiteLLM
```python
from gptcache import cache
from litellm.gpt_cache import completion # import completion from litellm.cache
import time

# Set your .env keys 
os.environ['OPENAI_API_KEY'] = ""
cache.init()
cache.set_openai_key()

question = "what's LiteLLM"
for _ in range(2):
    start_time = time.time()
    response = completion(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
```

