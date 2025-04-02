import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# High Availability Setup (Resolve DB Deadlocks)

Resolve any Database Deadlocks you see in high traffic by using this setup

## What causes the problem?

LiteLLM writes `UPDATE` and `UPSERT` queries to the DB. When using 10+ pods of LiteLLM, these queries can cause deadlocks since each pod could simultaneously attempt to update the same `user_id`, `team_id`, `key` etc. 

## How the high availability setup fixes the problem
- All pods will write to a Redis queue instead of the DB. 
- A single pod will acquire a lock on the DB and flush the redis queue to the DB. 


## How it works 

### Stage 1. Each pod writes updates to redis

Each pod will accumlate the spend updates for a key, user, team, etc and write the updates to a redis queue. 

<Image img={require('../../img/deadlock_fix_1.png')}  style={{ width: '900px', height: 'auto' }} />


### Stage 2. A single pod flushes the redis queue to the DB

A single pod will acquire a lock on the DB and flush all elements in the redis queue to the DB. 


<Image img={require('../../img/deadlock_fix_2.png')}  style={{ width: '900px', height: 'auto' }} />


## Setup
- Redis
- Postgres

```yaml showLineNumbers title="litellm proxy_config.yaml"
general_settings:
  use_redis_transaction_buffer: true

litellm_settings:
  cache: True
  cache_params:
    type: redis
    supported_call_types: []
```
