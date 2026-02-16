# Benchmark Guide Compliance Check

This document verifies that the AWS deployment matches the official [LiteLLM Benchmark Guide](https://docs.litellm.ai/docs/benchmarks) specifications.

## ✅ Compliance Summary

**Status:** FULLY COMPLIANT (as of commit f7de2ee + fixes)

All critical requirements from the benchmark guide have been implemented and verified.

## 📊 Detailed Comparison

### 1. Hardware Configuration

| Requirement | Benchmark Guide | Our Implementation | Status |
|-------------|----------------|--------------------| -------|
| CPU per instance | 4 cores | 4 vCPU (4096 units) | ✅ MATCH |
| RAM per instance | 8 GB | 8192 MB (8 GB) | ✅ MATCH |
| Number of instances | 4 | 4 (configurable) | ✅ MATCH |
| Workers per instance | 4 | 4 (configurable) | ✅ MATCH |
| **Total workers** | **16** | **16** | ✅ **MATCH** |

### 2. Database Configuration

| Requirement | Benchmark Guide | Our Implementation | Status |
|-------------|----------------|--------------------| -------|
| Database type | PostgreSQL | PostgreSQL 16.3 | ✅ MATCH |
| CPU cores | 4-8 cores | db.r6g.xlarge (4 vCPU) | ✅ MATCH |
| RAM | 16 GB | db.r6g.xlarge (32 GB) | ✅ EXCEEDS |
| Storage | 200 GB SSD | 200 GB gp3 | ✅ MATCH |
| IOPS | Not specified | 3000 IOPS | ✅ GOOD |
| Throughput | Not specified | 125 MBps | ✅ GOOD |
| Batch writes | `proxy_batch_write_at: 60` | `proxy_batch_write_at: 60` | ✅ MATCH |

**Note:** db.r6g.xlarge provides 32 GB RAM (exceeds 16 GB requirement), which is beneficial for performance.

### 3. Model Configuration

| Requirement | Benchmark Guide | Our Implementation | Status |
|-------------|----------------|--------------------| -------|
| Model name | `fake-openai-endpoint` | `fake-openai-endpoint` | ✅ MATCH |
| Model param | `openai/any` | `openai/any` | ✅ MATCH |
| API key | `test` | `test` | ✅ MATCH |
| API base | `https://exampleopenaiendpoint-production.up.railway.app/` | `https://exampleopenaiendpoint-production.up.railway.app/` | ✅ MATCH |

### 4. Load Testing Configuration (Locust)

| Requirement | Benchmark Guide | Our Implementation | Status |
|-------------|----------------|--------------------| -------|
| Concurrent users | 1,000 | 1,000 (configurable) | ✅ MATCH |
| Spawn rate | 500 users/sec | 500 users/sec | ✅ MATCH |
| Test duration | 5 minutes | 5 minutes (configurable) | ✅ MATCH |
| Wait time | 0.5-1 second | `between(0.5, 1)` | ✅ MATCH |

### 5. Expected Performance Targets

| Metric | Benchmark Guide (4 instances) | Our Target |
|--------|-------------------------------|------------|
| Median latency | ~100 ms | ~100 ms |
| P95 latency | ~150 ms | ~150 ms |
| P99 latency | ~240 ms | ~240 ms |
| Throughput | ~1,170 RPS | ~1,170 RPS |
| LiteLLM overhead | ~2 ms (median) | ~2 ms |

### 6. Configuration Settings

| Setting | Benchmark Guide | Our Implementation | Status |
|---------|----------------|--------------------| -------|
| `proxy_batch_write_at` | 60 | 60 | ✅ MATCH |
| `store_model_in_db` | true | true | ✅ MATCH |
| `master_key` | configured | via env var | ✅ MATCH |
| `database_url` | configured | via env var | ✅ MATCH |

### 7. Optional Features

| Feature | Benchmark Guide | Our Implementation | Status |
|---------|----------------|--------------------| -------|
| Redis cache | Optional (2-4 cores, 8GB RAM) | Not included | ⚠️ OPTIONAL |
| GCS logging | Tested, minimal impact | Not configured | ⚠️ OPTIONAL |
| LangSmith logging | Tested, minimal impact | Not configured | ⚠️ OPTIONAL |

**Note:** Redis is optional but recommended for production. It can reduce database load by 60-80%.

## 🔧 Changes Made to Achieve Compliance

### Initial Issues (Discovered)
1. ❌ Database too small (db.t3.medium vs. required 4-8 cores, 16GB RAM)
2. ❌ Missing `proxy_batch_write_at: 60` configuration
3. ❌ Wrong model parameter (`openai/fake` vs. `openai/any`)
4. ❌ Wrong Locust wait time (0.1-0.5s vs. 0.5-1s)
5. ❌ Storage too small (100 GB vs. 200 GB)

### Fixes Applied
1. ✅ Upgraded to db.r6g.xlarge (4 vCPU, 32 GB RAM)
2. ✅ Added `proxy_batch_write_at: 60` to config
3. ✅ Changed model to `openai/any`
4. ✅ Fixed Locust wait_time to `between(0.5, 1)`
5. ✅ Increased storage to 200 GB with 3000 IOPS

## 💰 Cost Impact

The benchmark-compliant configuration is more expensive due to the larger database:

| Component | Previous | Benchmark-Compliant | Monthly Cost Difference |
|-----------|----------|---------------------|-------------------------|
| Database | db.t3.medium | db.r6g.xlarge | +$150/month |
| Storage | 100 GB | 200 GB | +$12/month |
| **Total Impact** | | | **+$162/month** |

**New estimated monthly cost:** ~$600-620/month (vs. previous ~$440-460)

**For cost-conscious users:** The template now includes a `DBInstanceClass` parameter that allows choosing a smaller instance for non-benchmark testing:
- `db.t3.medium` - ~$60/month (sufficient for light testing)
- `db.r6g.xlarge` - ~$210/month (benchmark-compliant)

## 📝 Implementation Details

### CloudFormation Parameters

```yaml
Parameters:
  DBInstanceClass:
    Default: db.r6g.xlarge  # Benchmark-compliant
    AllowedValues:
      - db.t3.micro      # Dev/test only
      - db.t3.small      # Light load
      - db.t3.medium     # Moderate load
      - db.r6g.large     # High load
      - db.r6g.xlarge    # Benchmark spec (1-2K RPS)
      - db.r6g.2xlarge   # Very high load (>2K RPS)
```

### LiteLLM Config (SSM Parameter)

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/any
      api_key: test
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

general_settings:
  master_key: os.environ/PROXY_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true
  proxy_batch_write_at: 60  # Batch writes every 60 seconds
```

### Locust Configuration

```python
class LiteLLMUser(HttpUser):
    wait_time = between(0.5, 1)  # Matches benchmark guide

# Run with:
locust -f locustfile.py \
  --host=$LITELLM_HOST \
  --users=1000 \
  --spawn-rate=500 \
  --run-time=5m \
  --headless
```

## ✅ Verification Checklist

Before running benchmark tests, verify:

- [x] 4 ECS tasks running (4 vCPU, 8 GB RAM each)
- [x] 4 workers configured per task
- [x] Database is db.r6g.xlarge or equivalent
- [x] 200 GB storage allocated
- [x] `proxy_batch_write_at: 60` in config
- [x] `fake-openai-endpoint` model configured with `openai/any`
- [x] Locust test file uses `between(0.5, 1)` wait time
- [x] All targets healthy in load balancer
- [x] Database connections working

## 🎯 How to Deploy Benchmark-Compliant Configuration

### Default (Benchmark-Compliant)

```bash
./deploy.sh
# Uses db.r6g.xlarge by default
```

### Custom Database Size

```bash
aws cloudformation create-stack \
  --stack-name litellm-benchmark \
  --template-body file://cloudformation-ecs.yaml \
  --parameters \
    ParameterKey=DBPassword,ParameterValue=SecurePass123 \
    ParameterKey=MasterKey,ParameterValue=SecureMasterKey123456 \
    ParameterKey=DBInstanceClass,ParameterValue=db.r6g.xlarge \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

## 📚 References

- [Official Benchmark Guide](https://docs.litellm.ai/docs/benchmarks)
- [Measuring LiteLLM Overhead](https://docs.litellm.ai/docs/benchmarks#how-to-measure-litellm-overhead)
- [Database Configuration](https://docs.litellm.ai/docs/benchmarks#database-setup)
- [Locust Load Testing](https://docs.locust.io/)

## 🔄 Version History

- **v1.0** (commit 445c67c): Initial deployment (infrastructure only, no models)
- **v1.1** (commit 1cf0097): Added model configuration (db.t3.medium)
- **v2.0** (commit f7de2ee): Fixed model config, added SSM integration
- **v2.1** (current): **BENCHMARK COMPLIANT** - All specifications matched

---

**Last Verified:** 2026-02-16
**Benchmark Guide Version:** Current as of 2026-02
**Compliance Status:** ✅ FULLY COMPLIANT
