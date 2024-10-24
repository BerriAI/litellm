import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Life of a Request

## High Level architecture

<Image img={require('../../img/litellm_gateway.png')} style={{ width: '100%', maxWidth: '4000px' }} />


### Request Flow 

1. **User Sends Request**: The process begins when a user sends a request to the LiteLLM Proxy Server (Gateway).

2. [**Virtual Keys**](../virtual_keys): At this stage the `Bearer` token in the request is checked to ensure it is valid and under it's budget. [Here is the list of checks that run for each request](https://github.com/BerriAI/litellm/blob/ba41a72f92a9abf1d659a87ec880e8e319f87481/litellm/proxy/auth/auth_checks.py#L43)
    - 2.1 Check if the Virtual Key exists in Redis Cache or In Memory Cache
    - 2.2 **If not in Cache**, Lookup Virtual Key in DB

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
    - The `_PROXY_track_cost_callback` updates spend / usage in the LiteLLM database. [Here is everything tracked in the DB per request](https://github.com/BerriAI/litellm/blob/ba41a72f92a9abf1d659a87ec880e8e319f87481/schema.prisma#L172)
