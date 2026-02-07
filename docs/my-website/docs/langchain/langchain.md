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
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ['OPENAI_API_KEY'] = ""
chat = ChatLiteLLM(model="gpt-3.5-turbo")
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat.invoke(messages)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

```python
import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ['ANTHROPIC_API_KEY'] = ""
chat = ChatLiteLLM(model="claude-2", temperature=0.3)
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat.invoke(messages)
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```python
import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ['REPLICATE_API_TOKEN'] = ""
chat = ChatLiteLLM(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat.invoke(messages)
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```python
import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ['COHERE_API_KEY'] = ""
chat = ChatLiteLLM(model="command-nightly")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat.invoke(messages)
```

</TabItem>
</Tabs>

## Use Langchain ChatLiteLLM with MLflow

MLflow provides open-source observability solution for ChatLiteLLM.

To enable the integration, simply call `mlflow.litellm.autolog()` before in your code. No other setup is necessary.

```python
import mlflow

mlflow.litellm.autolog()
```

Once the auto-tracing is enabled, you can invoke `ChatLiteLLM` and see recorded traces in MLflow.

```python
import os
from langchain.chat_models import ChatLiteLLM

os.environ['OPENAI_API_KEY']="sk-..."

chat = ChatLiteLLM(model="gpt-4o-mini")
chat.invoke("Hi!")
```

## Use Langchain ChatLiteLLM with Lunary
```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.schema import HumanMessage
import litellm

os.environ["LUNARY_PUBLIC_KEY"] = "" # from https://app.lunary.ai/settings
os.environ['OPENAI_API_KEY']="sk-..."

litellm.success_callback = ["lunary"] 
litellm.failure_callback = ["lunary"] 

chat = ChatLiteLLM(
  model="gpt-4o"
  messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```

Get more details [here](../observability/lunary_integration.md)

## Use LangChain ChatLiteLLM + Langfuse
Checkout this section [here](../observability/langfuse_integration#use-langchain-chatlitellm--langfuse) for more details on how to integrate Langfuse with ChatLiteLLM.

## Using Tags with LangChain and LiteLLM

Tags are a powerful feature in LiteLLM that allow you to categorize, filter, and track your LLM requests. When using LangChain with LiteLLM, you can pass tags through the `extra_body` parameter in the metadata.

### Basic Tag Usage

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ['OPENAI_API_KEY'] = "sk-your-key-here"

chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    extra_body={
        "metadata": {
            "tags": ["production", "customer-support", "high-priority"]
        }
    }
)

messages = [
    SystemMessage(content="You are a helpful customer support assistant."),
    HumanMessage(content="How do I reset my password?")
]

response = chat.invoke(messages)
print(response)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ['ANTHROPIC_API_KEY'] = "sk-ant-your-key-here"

chat = ChatOpenAI(
    model="claude-3-sonnet-20240229",
    temperature=0.7,
    extra_body={
        "metadata": {
            "tags": ["research", "analysis", "claude-model"]
        }
    }
)

messages = [
    SystemMessage(content="You are a research analyst."),
    HumanMessage(content="Analyze this market trend...")
]

response = chat.invoke(messages)
print(response)
```

</TabItem>

<TabItem value="litellm-proxy" label="LiteLLM Proxy">

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# No API key needed when using proxy
chat = ChatOpenAI(
    openai_api_base="http://localhost:4000",  # Your proxy URL
    model="gpt-4o",
    temperature=0.7,
    extra_body={
        "metadata": {
            "tags": ["proxy", "team-alpha", "feature-flagged"],
            "generation_name": "customer-onboarding",
            "trace_user_id": "user-12345"
        }
    }
)

messages = [
    SystemMessage(content="You are an onboarding assistant."),
    HumanMessage(content="Welcome our new customer!")
]

response = chat.invoke(messages)
print(response)
```

</TabItem>
</Tabs>

### Advanced Tag Patterns

#### Dynamic Tags Based on Context

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def create_chat_with_tags(user_type: str, feature: str):
    """Create a chat instance with dynamic tags based on context"""
    
    # Build tags dynamically
    tags = ["langchain-integration"]
    
    if user_type == "premium":
        tags.extend(["premium-user", "high-priority"])
    elif user_type == "enterprise":
        tags.extend(["enterprise", "custom-sla"])
    else:
        tags.append("standard-user")
    
    # Add feature-specific tags
    if feature == "code-review":
        tags.extend(["development", "code-analysis"])
    elif feature == "content-gen":
        tags.extend(["marketing", "content-creation"])
    
    return ChatOpenAI(
        openai_api_base="http://localhost:4000",
        model="gpt-4o",
        temperature=0.7,
        extra_body={
            "metadata": {
                "tags": tags,
                "user_type": user_type,
                "feature": feature,
                "trace_user_id": f"user-{user_type}-{feature}"
            }
        }
    )

# Usage examples
premium_chat = create_chat_with_tags("premium", "code-review")
enterprise_chat = create_chat_with_tags("enterprise", "content-gen")

messages = [HumanMessage(content="Help me with this task")]
response = premium_chat.invoke(messages)
```

#### Tags for Cost Tracking and Analytics

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Tags for cost tracking
cost_tracking_chat = ChatOpenAI(
    openai_api_base="http://localhost:4000",
    model="gpt-4o",
    temperature=0.7,
    extra_body={
        "metadata": {
            "tags": [
                "cost-center-marketing",
                "budget-q4-2024",
                "project-launch-campaign",
                "high-cost-model"  # Flag for expensive models
            ],
            "department": "marketing",
            "project_id": "campaign-2024-q4",
            "cost_threshold": "high"
        }
    }
)

messages = [
    SystemMessage(content="You are a marketing copywriter."),
    HumanMessage(content="Create compelling ad copy for our new product launch.")
]

response = cost_tracking_chat.invoke(messages)
```

#### Tags for A/B Testing

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import random

def create_ab_test_chat(test_variant: str = None):
    """Create chat instance for A/B testing with appropriate tags"""
    
    if test_variant is None:
        test_variant = random.choice(["variant-a", "variant-b"])
    
    return ChatOpenAI(
        openai_api_base="http://localhost:4000",
        model="gpt-4o",
        temperature=0.7 if test_variant == "variant-a" else 0.9,  # Different temp for variants
        extra_body={
            "metadata": {
                "tags": [
                    "ab-test-experiment-1",
                    f"variant-{test_variant}",
                    "temperature-test",
                    "user-experience"
                ],
                "experiment_id": "ab-test-001",
                "variant": test_variant,
                "test_group": "temperature-optimization"
            }
        }
    )

# Run A/B test
variant_a_chat = create_ab_test_chat("variant-a")
variant_b_chat = create_ab_test_chat("variant-b")

test_message = [HumanMessage(content="Explain quantum computing in simple terms")]

response_a = variant_a_chat.invoke(test_message)
response_b = variant_b_chat.invoke(test_message)
```

### Tag Best Practices

#### 1. **Consistent Naming Convention**
```python
# ✅ Good: Consistent, descriptive tags
tags = ["production", "api-v2", "customer-support", "urgent"]

# ❌ Avoid: Inconsistent or unclear tags
tags = ["prod", "v2", "support", "urgent123"]
```

#### 2. **Hierarchical Tags**
```python
# ✅ Good: Hierarchical structure
tags = ["env:production", "team:backend", "service:api", "priority:high"]

# This allows for easy filtering and grouping
```

#### 3. **Include Context Information**
```python
extra_body={
    "metadata": {
        "tags": ["production", "user-onboarding"],
        "user_id": "user-12345",
        "session_id": "session-abc123",
        "feature_flag": "new-onboarding-flow",
        "environment": "production"
    }
}
```

#### 4. **Tag Categories**
Consider organizing tags into categories:
- **Environment**: `production`, `staging`, `development`
- **Team/Service**: `backend`, `frontend`, `api`, `worker`
- **Feature**: `authentication`, `payment`, `notification`
- **Priority**: `critical`, `high`, `medium`, `low`
- **User Type**: `premium`, `enterprise`, `free`

### Using Tags with LiteLLM Proxy

When using tags with LiteLLM Proxy, you can:

1. **Filter requests** based on tags
2. **Track costs** by tags in spend reports
3. **Apply routing rules** based on tags
4. **Monitor usage** with tag-based analytics

#### Example Proxy Configuration with Tags

```yaml
# config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: your-key

# Tag-based routing rules
tag_routing:
  - tags: ["premium", "high-priority"]
    models: ["gpt-4o", "claude-3-opus"]
  - tags: ["standard"]
    models: ["gpt-3.5-turbo", "claude-3-haiku"]
```

### Monitoring and Analytics

Tags enable powerful analytics capabilities:

```python
# Example: Get spend reports by tags
import requests

response = requests.get(
    "http://localhost:4000/global/spend/report",
    headers={"Authorization": "Bearer sk-your-key"},
    params={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "group_by": "tags"
    }
)

spend_by_tags = response.json()
```

This documentation covers the essential patterns for using tags effectively with LangChain and LiteLLM, enabling better organization, tracking, and analytics of your LLM requests.
