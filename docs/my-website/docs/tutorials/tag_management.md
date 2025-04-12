import Image from '@theme/IdealImage';

# [Beta] Routing based on request metadata

Create routing rules based on request metadata.

## Setup

Add the following to your litellm proxy config yaml file.

```yaml showLineNumbers title="litellm proxy config.yaml"
router_settings:
  enable_tag_filtering: True # 👈 Key Change
```

## 1. Create a tag

On the LiteLLM UI, navigate to Experimental > Tag Management > Create Tag.

Create a tag called `private-data` and only select the allowed models for requests with this tag. Once created, you will see the tag in the Tag Management page.

<Image img={require('../../img/tag_create.png')}  style={{ width: '800px', height: 'auto' }} />


## 2. Test Tag Routing

Now we will test the tag based routing rules.

### 2.1 Invalid model

This request will fail since we send `tags=private-data` but the model `gpt-4o` is not in the allowed models for the `private-data` tag.

<Image img={require('../../img/tag_invalid.png')}  style={{ width: '800px', height: 'auto' }} />

<br />

### 2.2 Valid model

This request will succeed since we send `tags=private-data` and the model `us.anthropic.claude-3-7-sonnet-20250219-v1:0` is in the allowed models for the `private-data` tag.

<Image img={require('../../img/tag_valid.png')}  style={{ width: '800px', height: 'auto' }} />



