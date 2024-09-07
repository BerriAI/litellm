import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Life of a Request

## High Level architecture

<Image img={require('../../img/litellm_gateway.png')} />


### Request Flow 

1. **User Sends Request**: The process begins when a user sends a request to the LiteLLM Proxy Server (Gateway).

2. [**Virtual Keys**](../virtual_keys): The request first passes through the Virtual Keys component

3. **Rate Limiting**: The MaxParallelRequestsHandler applies rate limiting to manage the flow of requests.

4. **Proxy Server Processing**: The request is then processed by the LiteLLM proxy_server.py, which handles the core logic of the proxy.

5. [**LiteLLM Router**](../routing): LiteLLM Router**: The LiteLLM Router determines where to send the request based on the configuration and request parameters.

6. **Model Interaction**: The request is sent to the appropriate model API (litellm.completion() or litellm.embedding()) for processing.

7. **Response**: The model's response is sent back through the same components to the user.

8. **Post-Request Processing**: After the response is sent, several asynchronous operations occur:
   - The _PROXY_track_cost_callback updates spend in the database.
   - Logging to LangFuse for analytics and monitoring.
   - The MaxParallelRequestsHandler updates virtual key usage and performs post-request cleanup.
