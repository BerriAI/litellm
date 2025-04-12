import Image from '@theme/IdealImage';

# Routing based on request metadata

Create routing rules based on request metadata.

## Setup

Add the following to your litellm proxy config yaml file.
```yaml
router_settings:
  enable_tag_filtering: True # 👈 Key Change
```

## 1. Create a tag



## 2. Test Tag Routing

### 2.1 Invalid model

<Image img={require('../../img/tag_invalid.png')}  style={{ width: '800px', height: 'auto' }} />

### 2.2 Valid model

<Image img={require('../../img/tag_valid.png')}  style={{ width: '800px', height: 'auto' }} />



