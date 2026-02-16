# AWS Deployment Files - Index

Complete 1-click deployment solution for LiteLLM on AWS, configured to match [benchmark specifications](https://docs.litellm.ai/docs/benchmarks).

## 📋 Quick Reference

| File | Purpose | Use When |
|------|---------|----------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute deployment guide | You want to get started immediately |
| [README.md](README.md) | Complete documentation | You need detailed instructions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture deep-dive | You want to understand the design |
| [cloudformation-ecs.yaml](cloudformation-ecs.yaml) | Infrastructure template | Deploying via CloudFormation |
| [deploy.sh](deploy.sh) | Automated deployment script | You want true 1-click deployment |
| [test-deployment.sh](test-deployment.sh) | Validation and testing | After deployment to verify setup |
| [locustfile.py](locustfile.py) | Load testing script | Running benchmark tests |
| [cost-calculator.sh](cost-calculator.sh) | Cost estimation tool | Planning your budget |
| [example-config.yaml](example-config.yaml) | LiteLLM configuration example | Customizing your deployment |

## 🚀 Getting Started

### Option 1: Fastest (1-Click Script)

```bash
cd deploy/aws
./deploy.sh
```

### Option 2: CloudFormation CLI

```bash
aws cloudformation create-stack \
  --stack-name litellm-benchmark \
  --template-body file://cloudformation-ecs.yaml \
  --parameters \
    ParameterKey=DBPassword,ParameterValue=YourPassword123 \
    ParameterKey=MasterKey,ParameterValue=YourMasterKey12345678 \
  --capabilities CAPABILITY_IAM
```

### Option 3: AWS Console

1. Go to CloudFormation in AWS Console
2. Create Stack → Upload template file
3. Use `cloudformation-ecs.yaml`
4. Fill in parameters
5. Create stack

## 📊 Benchmark Configuration

**What You Get:**
```
4 ECS Tasks (Fargate)
├── 4 vCPU per task
├── 8 GB RAM per task
├── 4 workers per task
└── Total: 16 vCPU, 32 GB RAM, 16 workers

PostgreSQL Database (RDS)
├── db.t3.medium
├── 2 vCPU, 4 GB RAM
└── 100 GB storage

Application Load Balancer
└── HTTP/HTTPS with health checks
```

**Expected Performance:**
- **Median latency:** ~100 ms
- **P95 latency:** ~150 ms
- **P99 latency:** ~240 ms
- **Throughput:** ~1,170 RPS
- **LiteLLM overhead:** ~2 ms

## 📁 File Descriptions

### QUICKSTART.md
Quick start guide for deploying in under 5 minutes. Includes:
- Prerequisites
- Deployment steps
- First API call
- Cleanup instructions

**Read this if:** You want to deploy quickly without details.

### README.md
Complete deployment documentation covering:
- Detailed deployment options
- Testing procedures
- Monitoring and troubleshooting
- Cost optimization
- Advanced configuration

**Read this if:** You need comprehensive documentation.

### ARCHITECTURE.md
In-depth architecture documentation including:
- Network architecture diagrams
- Security group configuration
- Component descriptions
- Data flow diagrams
- High availability design
- Scaling strategies

**Read this if:** You want to understand how everything works.

### cloudformation-ecs.yaml
CloudFormation Infrastructure-as-Code template that creates:
- VPC with public/private subnets
- Application Load Balancer
- ECS Fargate cluster and service
- RDS PostgreSQL database
- Security groups
- IAM roles
- Secrets Manager secrets

**Use this if:** Deploying via CloudFormation.

### deploy.sh
Automated deployment script that:
- Validates prerequisites
- Prompts for required parameters
- Creates CloudFormation stack
- Waits for completion
- Displays endpoints and credentials
- Runs basic health checks

**Use this if:** You want the easiest deployment experience.

### test-deployment.sh
Validation script that checks:
- ECS service status
- Task configuration
- Health endpoints
- API response time
- Database status
- Load balancer health

**Use this if:** You want to verify your deployment.

### locustfile.py
Locust load testing script for:
- Replicating benchmark tests
- Custom load testing scenarios
- Measuring latency and throughput
- Tracking LiteLLM overhead

**Use this if:** You want to benchmark your deployment.

### cost-calculator.sh
Interactive cost estimation tool that:
- Calculates monthly costs
- Shows cost breakdown
- Estimates savings with reserved capacity
- Compares alternative configurations

**Use this if:** You need cost estimates before deploying.

### example-config.yaml
LiteLLM proxy configuration example showing:
- Multiple LLM provider setup
- Router configuration
- Caching options
- Monitoring integrations
- Rate limiting
- Team management

**Use this if:** You want to customize LiteLLM configuration.

## 🎯 Common Workflows

### 1. Deploy and Test

```bash
# Deploy
./deploy.sh

# Wait for completion (script handles this)

# Test deployment
./test-deployment.sh

# Run benchmark
export LITELLM_MASTER_KEY="your-master-key"
export LITELLM_HOST="http://your-alb-url"
pip install locust
locust -f locustfile.py --users=1000 --spawn-rate=500 --run-time=5m --headless
```

### 2. Estimate Costs

```bash
# Calculate costs before deploying
./cost-calculator.sh

# Enter your configuration:
# - Number of tasks: 4
# - vCPU per task: 4
# - Memory per task: 8
# - RDS instance: t3.medium
```

### 3. Customize Configuration

```bash
# 1. Copy example config
cp example-config.yaml my-config.yaml

# 2. Edit with your API keys and settings
nano my-config.yaml

# 3. Update CloudFormation template to mount config
# (See README.md for detailed instructions)

# 4. Redeploy
aws cloudformation update-stack ...
```

### 4. Scale Your Deployment

```bash
# Scale to 8 tasks
aws ecs update-service \
  --cluster litellm-benchmark-LiteLLM-Cluster \
  --service litellm-benchmark-litellm-service \
  --desired-count 8

# Or redeploy with new parameters
DESIRED_TASKS=8 ./deploy.sh
```

### 5. Monitor and Troubleshoot

```bash
# View logs
aws logs tail /ecs/litellm-benchmark-litellm --follow

# Check service status
aws ecs describe-services \
  --cluster litellm-benchmark-LiteLLM-Cluster \
  --services litellm-benchmark-litellm-service

# View CloudWatch metrics
# Go to CloudWatch Console → Metrics → ECS/RDS/ALB
```

### 6. Cleanup

```bash
# Delete entire stack
aws cloudformation delete-stack --stack-name litellm-benchmark

# Verify deletion
aws cloudformation describe-stacks --stack-name litellm-benchmark
```

## 💰 Cost Summary

**Monthly Cost (Pay-as-you-go):** ~$440-460

**Breakdown:**
- ECS Fargate: ~$350
- RDS PostgreSQL: ~$60
- ALB: ~$24
- NAT Gateway: ~$33
- Data Transfer: ~$10-30
- Other (Secrets, Logs): ~$9

**With Reserved Capacity (1-year):** ~$350-370/month
**With Reserved Capacity (3-year):** ~$270-290/month

Run `./cost-calculator.sh` for detailed estimates.

## 🏗️ Architecture Summary

```
Internet
   ↓
Application Load Balancer (Public)
   ↓
ECS Tasks (Private) × 4
   └─ 4 vCPU, 8 GB RAM, 4 workers each
   ↓
RDS PostgreSQL (Private)
   └─ db.t3.medium, 100 GB
```

**Security:**
- Tasks in private subnets
- Database not publicly accessible
- Security groups with least privilege
- Secrets in Secrets Manager

**High Availability:**
- Multi-AZ deployment
- Auto-scaling capability
- Health check monitoring
- Automatic task replacement

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## 📈 Performance Benchmarks

### Benchmark Test Results

Using Locust with 1,000 concurrent users:

| Metric | 2 Instances | 4 Instances (Target) |
|--------|-------------|----------------------|
| Median Latency | 200 ms | **100 ms** |
| P95 Latency | 630 ms | **150 ms** |
| P99 Latency | 1,200 ms | **240 ms** |
| Average Latency | 262 ms | **111.7 ms** |
| Throughput | 1,035 RPS | **1,170 RPS** |
| LiteLLM Overhead | 12 ms | **2 ms** |

**Key Finding:** Doubling instances from 2 to 4 halves median latency.

## 🔧 Configuration Options

### Environment Variables

Set in ECS task definition:
- `DATABASE_URL` - PostgreSQL connection (auto-configured)
- `STORE_MODEL_IN_DB` - Enable model management
- `PROXY_MASTER_KEY` - API authentication key
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `REDIS_HOST` - Redis cache host (optional)

### Task Parameters

Adjustable via CloudFormation parameters:
- `DesiredTaskCount` - Number of ECS tasks (default: 4)
- `NumWorkersPerTask` - Workers per task (default: 4)
- `TaskCPU` - CPU units per task (default: 4096)
- `TaskMemory` - Memory MB per task (default: 8192)

### Database Settings

Adjustable for performance:
- Instance class (t3.micro → r6g.large)
- Storage size (100 GB → 1000 GB)
- Multi-AZ for high availability
- Read replicas for read-heavy workloads

## 📚 Additional Resources

### Documentation
- [LiteLLM Docs](https://docs.litellm.ai)
- [Benchmark Guide](https://docs.litellm.ai/docs/benchmarks)
- [Proxy Configuration](https://docs.litellm.ai/docs/proxy/configs)

### AWS Documentation
- [ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [RDS Performance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)
- [Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

### Support
- [GitHub Issues](https://github.com/BerriAI/litellm/issues)
- [Community Discussions](https://github.com/BerriAI/litellm/discussions)

## ✅ Checklist

Before deploying:
- [ ] AWS CLI installed and configured
- [ ] Appropriate AWS permissions
- [ ] Generated strong database password (min 8 chars)
- [ ] Generated strong master key (min 16 chars)
- [ ] Reviewed cost estimates
- [ ] Selected appropriate AWS region

After deploying:
- [ ] Verify all tasks are running
- [ ] Test health endpoints
- [ ] Make test API call
- [ ] Run validation script
- [ ] Set up monitoring/alerting
- [ ] Configure API keys for real LLM providers
- [ ] Run benchmark tests
- [ ] Document your endpoints and credentials

For production:
- [ ] Enable HTTPS with SSL certificate
- [ ] Configure custom domain
- [ ] Enable auto-scaling
- [ ] Set up CloudWatch alarms
- [ ] Implement backup strategy
- [ ] Review security best practices
- [ ] Enable CloudTrail for auditing
- [ ] Consider Multi-AZ RDS
- [ ] Evaluate reserved capacity savings

## 🤝 Contributing

Found an issue or want to improve these deployment templates?
- Open an issue: https://github.com/BerriAI/litellm/issues
- Submit a PR: https://github.com/BerriAI/litellm/pulls

## 📝 License

These deployment templates are part of the LiteLLM project.
See the main repository for license information.

---

**Last Updated:** February 2026
**LiteLLM Version:** Compatible with main-latest
**Benchmark Reference:** https://docs.litellm.ai/docs/benchmarks
