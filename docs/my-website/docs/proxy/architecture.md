import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Life of a Request

## High Level architecture

<Image img={require('../../img/litellm_gateway.png')} />


### Request Flow 

1. **User Sends Request**: The process begins when a user sends a request to the LiteLLM Proxy Server (Gateway).

2. [**Virtual Keys**](../virtual_keys): At this stage the `Bearer` token in the request is checked to ensure it is valid and under it's budget

3. **Rate Limiting**: The [MaxParallelRequestsHandler](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/hooks/parallel_request_limiter.py) checks the **rate limit (rpm/tpm)** for the the following components:
    - Global Server Rate Limit
    - Virtual Key Rate Limit
    - User Rate Limit
    - Team Limit

4. **LiteLLM `proxy_server.py`**: Contains the `/chat/completions` and `/embeddings` endpoints. Requests to these endpoints are sent through the LiteLLM Router

5. [**LiteLLM Router**](../routing): The LiteLLM Router handles Load balancing, Fallbacks, Retries for LLM API deployments.

6. [**litellm.completion() / litellm.embedding()**:](../index#litellm-python-sdk) The litellm Python SDK is used to call the LLM in the OpenAI API format (Translation and parameter mapping)

7. **Post-Request Processing**: After the response is sent back to the client, the following **asynchronous** tasks are performed:
   - [Logging to LangFuse (logging destination is configurable)](./logging)
   - The [MaxParallelRequestsHandler](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/hooks/parallel_request_limiter.py) updates the rpm/tpm usage for the 
        - Global Server Rate Limit
        - Virtual Key Rate Limit
        - User Rate Limit
        - Team Limit
    - The `_PROXY_track_cost_callback` updates spend / usage in the LiteLLM database.
