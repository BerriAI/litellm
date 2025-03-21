import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Image URL Handling 

<Image img={require('../../img/image_handling.png')}  style={{ width: '900px', height: 'auto' }} />

Some LLM API's don't support url's for images, but do support base-64 strings. 

For those, LiteLLM will:

1. Detect a URL being passed
2. Check if the LLM API supports a URL
3. Else, will download the base64 
4. Send the provider a base64 string. 


LiteLLM also caches this result, in-memory to reduce latency for subsequent calls. 

The limit for an in-memory cache is 1MB. 