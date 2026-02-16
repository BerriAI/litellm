# Known Issues & Limitations

## ⚠️ Benchmark Testing Limitation

### Issue
The current CloudFormation deployment **cannot run the full benchmark test** as documented in the [LiteLLM Benchmark Guide](https://docs.litellm.ai/docs/benchmarks).

### Root Cause
The deployment does not configure the required `fake-openai-endpoint` model needed for benchmark testing.

### What's Missing

The benchmark guide requires this model configuration:

```yaml
model_list:
  - model_name: "fake-openai-endpoint"
    litellm_params:
      model: openai/any
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      api_key: "test"
```

**Current State:**
- ❌ No `config.yaml` mounted to ECS tasks
- ❌ No model list configured
- ❌ Environment variables only (DATABASE_URL, STORE_MODEL_IN_DB)

**Result:**
- API calls fail with: `Invalid model name passed in model=fake-openai-endpoint`
- Benchmark tests cannot run
- Only infrastructure health endpoints work

### What Works

Despite this limitation, the deployment successfully:
- ✅ Creates all infrastructure (VPC, ECS, RDS, ALB)
- ✅ Runs 4 ECS tasks with correct resource allocation (4 vCPU, 8 GB RAM)
- ✅ Configures 4 workers per task (16 total workers)
- ✅ Health endpoints respond correctly
- ✅ Database connections work
- ✅ Load balancer routes traffic properly

### Impact

**Cannot Measure:**
- Actual API request latency
- LiteLLM overhead (via `x-litellm-overhead-duration-ms` header)
- Requests per second (RPS) under load
- P50/P95/P99 latency percentiles for API calls

**Can Measure:**
- Infrastructure health endpoint latency (~600-800ms observed)
- Basic connectivity and routing
- Resource allocation and task health

### Workarounds

#### Option 1: Use UI to Add Models (Partial)
After deployment, use the LiteLLM admin UI to add models via the `/model/new` endpoint. However, this requires:
- Navigating to the Load Balancer URL
- Using the master key for authentication
- Manually adding each model through the UI
- **Limitation:** The UI doesn't support adding the fake endpoint easily

#### Option 2: Update via API (Partial)
Use the `/model/new` endpoint to programmatically add models:

```bash
curl -X POST "http://<load-balancer-url>/model/new" \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fake-openai-endpoint",
    "litellm_params": {
      "model": "openai/any",
      "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
      "api_key": "test"
    }
  }'
```

**Limitation:** Models added this way are stored in the database but may not persist correctly across deployments.

#### Option 3: Manual Config File Mount (Complex)
Manually create and mount a config file:
1. Create `config.yaml` with model configuration
2. Upload to S3 or store in Secrets Manager
3. Update ECS task definition to mount the file
4. Redeploy ECS service

**Limitation:** Requires manual CloudFormation template modification and understanding of ECS volume mounts.

### Recommended Fix

To properly support benchmark testing, the deployment should:

1. **Add ConfigMap Support**
   ```yaml
   # In CloudFormation template
   ConfigFile:
     Type: AWS::SSM::Parameter
     Properties:
       Name: !Sub /${AWS::StackName}/litellm-config
       Type: String
       Value: !Sub |
         model_list:
           - model_name: "fake-openai-endpoint"
             litellm_params:
               model: openai/any
               api_base: https://exampleopenaiendpoint-production.up.railway.app/
               api_key: "test"
   ```

2. **Mount Config to ECS Task**
   ```yaml
   # In TaskDefinition
   Command:
     - '--config'
     - '/app/config.yaml'
     - '--port'
     - '4000'
     - '--num_workers'
     - !Ref NumWorkersPerTask

   # Add volume and mount
   Volumes:
     - Name: config
       # Use EFS or bind mount from Parameter Store
   MountPoints:
     - SourceVolume: config
       ContainerPath: /app/config.yaml
       ReadOnly: true
   ```

3. **Or Use Environment Variables for Config**
   LiteLLM supports setting config via `LITELLM_CONFIG` environment variable:
   ```yaml
   Environment:
     - Name: LITELLM_CONFIG
       Value: !Sub |
         {"model_list": [{"model_name": "fake-openai-endpoint", "litellm_params": {"model": "openai/any", "api_base": "https://exampleopenaiendpoint-production.up.railway.app/", "api_key": "test"}}]}
   ```

### Testing Performed

**Successful Tests:**
- ✅ Infrastructure deployment (all resources created)
- ✅ ECS service health (4/4 tasks running)
- ✅ Load balancer health (4/4 targets healthy)
- ✅ Database connectivity
- ✅ Health endpoint responses (HTTP 200)

**Failed Tests:**
- ❌ Locust benchmark with 1,000 users (no models configured)
- ❌ API `/v1/chat/completions` requests (invalid model error)
- ❌ LiteLLM overhead measurement (no successful API calls)

**Observed Metrics:**
- Health endpoint latency: 600-800ms (higher than expected)
- Task startup time: ~2-3 minutes
- Database connection time: ~8 minutes (RDS creation)

### Related Issues

- [LiteLLM Benchmark Guide](https://docs.litellm.ai/docs/benchmarks) - Official benchmark documentation
- CloudFormation deployment uses environment variables only, no config file mounting
- Locust test file (`locustfile.py`) correctly references `fake-openai-endpoint`

### Status

**Status:** Known Limitation
**Severity:** High (prevents primary use case - benchmark testing)
**Workaround:** Manual model configuration via API (not tested)
**Fix Required:** Update CloudFormation template to mount config file

### Timeline

- **Discovered:** 2026-02-16
- **During:** Initial benchmark deployment test
- **Stack Deleted:** 2026-02-16 (to save costs)

### Next Steps

For anyone wanting to run the full benchmark:

1. **Fix the template** by adding config file mounting
2. **Test with fake endpoint** to verify model configuration
3. **Run full Locust benchmark** with 1,000 users
4. **Measure LiteLLM overhead** via response headers
5. **Compare results** with benchmark guide expectations:
   - Median latency: ~100 ms
   - P95 latency: ~150 ms
   - Throughput: ~1,170 RPS
   - LiteLLM overhead: ~2 ms

### References

- [LiteLLM Benchmark Documentation](https://docs.litellm.ai/docs/benchmarks)
- [How to Measure LiteLLM Overhead](https://docs.litellm.ai/docs/benchmarks#how-to-measure-litellm-overhead)
- [LiteLLM Proxy Configuration](https://docs.litellm.ai/docs/proxy/configs)
- [CloudFormation ECS Task Definition](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ecs-taskdefinition.html)

---

**Last Updated:** 2026-02-16
**Tested By:** Deployment validation and benchmark attempt
**Cost Impact:** ~$15/day while running (~$0.60/hour)
