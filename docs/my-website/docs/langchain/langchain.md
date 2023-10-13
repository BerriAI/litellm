import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using ChatLiteLLM() - Langchain

## Pre-Requisites
```shell
!pip install litellm langchain
```
## Quick Start

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ['OPENAI_API_KEY'] = ""
chat = ChatLiteLLM(model="gpt-3.5-turbo")
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ['ANTHROPIC_API_KEY'] = ""
chat = ChatLiteLLM(model="claude-2", temperature=0.3)
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ['REPLICATE_API_TOKEN'] = ""
chat = ChatLiteLLM(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat(messages)
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ['COHERE_API_KEY'] = ""
chat = ChatLiteLLM(model="command-nightly")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat(messages)
```

</TabItem>
<TabItem value="palm" label="PaLM - Google">

```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ['PALM_API_KEY'] = ""
chat = ChatLiteLLM(model="palm/chat-bison")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat(messages)
```
</TabItem>
</Tabs>
