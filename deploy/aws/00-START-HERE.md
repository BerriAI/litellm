# 🚀 LiteLLM AWS Benchmark Deployment - START HERE

## ✅ What You've Got

A complete, production-ready 1-click deployment solution for LiteLLM on AWS, configured exactly as specified in the [benchmark guide](https://docs.litellm.ai/docs/benchmarks).

## ✅ Ready for Full Benchmark Testing

This deployment includes complete model configuration and is **ready to run the official LiteLLM benchmarks** right out of the box!

**Included:**
- ✅ Pre-configured `fake-openai-endpoint` model
- ✅ Automatic config management via AWS SSM
- ✅ Full API functionality for benchmark testing
- ✅ LiteLLM overhead measurement support

**What works:** Infrastructure, model configuration, API requests, benchmark testing, performance measurement

👉 **See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for historical notes on earlier limitations (now resolved).**

## 🎯 Benchmark Configuration

This deployment creates:

```
┌─────────────────────────────────────────────┐
│  4 ECS Tasks (Fargate)                      │
│  ├─ 4 vCPU per task                         │
│  ├─ 8 GB RAM per task                       │
│  └─ 4 workers per task                      │
│                                              │
│  = 16 vCPU, 32 GB RAM, 16 workers total    │
└─────────────────────────────────────────────┘

Expected Performance:
✓ Median latency: ~100 ms
✓ P95 latency: ~150 ms
✓ P99 latency: ~240 ms
✓ Throughput: ~1,170 RPS
✓ LiteLLM overhead: ~2 ms
```

## 📁 Files Overview

| File | What It Does |
|------|--------------|
| **[QUICKSTART.md](QUICKSTART.md)** | Deploy in 5 minutes ⚡ |
| **[deploy.sh](deploy.sh)** | Automated deployment script 🤖 |
| **[README.md](README.md)** | Complete documentation 📖 |
| **[test-deployment.sh](test-deployment.sh)** | Verify your deployment ✅ |
| **[locustfile.py](locustfile.py)** | Run benchmark tests 📊 |
| **[cost-calculator.sh](cost-calculator.sh)** | Estimate costs 💰 |
| **[cloudformation-ecs.yaml](cloudformation-ecs.yaml)** | Infrastructure template ☁️ |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Deep dive into architecture 🏗️ |
| **[INDEX.md](INDEX.md)** | Complete file index 📋 |
| **[example-config.yaml](example-config.yaml)** | Configuration example ⚙️ |

## 🏃 Quick Deploy (2 minutes)

```bash
# 1. Navigate to this directory
cd deploy/aws

# 2. Run the deployment script
./deploy.sh

# 3. Enter your credentials when prompted
#    - Database password (min 8 chars)
#    - Master key (min 16 chars)

# 4. Wait ~10-15 minutes for deployment
# 5. Copy your API endpoint and master key when done!
```

## 🧪 Test Your Deployment

After deployment completes:

```bash
# Run validation tests
./test-deployment.sh

# Install Locust for load testing
pip install locust

# Run benchmark test (replicates the benchmark guide)
export LITELLM_MASTER_KEY="your-master-key-from-deployment"
export LITELLM_HOST="http://your-alb-url"

locust -f locustfile.py \
  --host=$LITELLM_HOST \
  --users=1000 \
  --spawn-rate=500 \
  --run-time=5m \
  --headless
```

## 💰 Cost Estimate

**Monthly Cost:** ~$440-460 (pay-as-you-go)

Run the cost calculator for detailed breakdown:
```bash
./cost-calculator.sh
```

**Savings with Reserved Capacity:**
- 1-year: ~$350-370/month (20-25% savings)
- 3-year: ~$270-290/month (40-45% savings)

## 📖 Documentation Guide

### New to AWS or LiteLLM?
→ Start with **[QUICKSTART.md](QUICKSTART.md)**

### Want detailed instructions?
→ Read **[README.md](README.md)**

### Want to understand the architecture?
→ Study **[ARCHITECTURE.md](ARCHITECTURE.md)**

### Need cost estimates?
→ Run **[cost-calculator.sh](cost-calculator.sh)**

### Ready to deploy?
→ Run **[deploy.sh](deploy.sh)**

### Want to verify deployment?
→ Run **[test-deployment.sh](test-deployment.sh)**

### Need to customize configuration?
→ See **[example-config.yaml](example-config.yaml)**

## ⚙️ What Gets Created

### Network Layer
- VPC with public and private subnets
- Internet Gateway and NAT Gateway
- Security Groups for ALB, ECS, and RDS
- Route tables

### Compute Layer
- ECS Fargate cluster with 4 tasks
- Application Load Balancer
- Auto-scaling configuration (optional)

### Data Layer
- RDS PostgreSQL database (db.t3.medium)
- Automated backups
- Encrypted storage

### Security
- Secrets Manager for sensitive data
- IAM roles with least privilege
- Private subnets for compute and data

### Monitoring
- CloudWatch Logs for ECS tasks
- CloudWatch Metrics for all services
- Health check endpoints

## 🎛️ Customization Options

### Scale to 8 instances
```bash
DESIRED_TASKS=8 ./deploy.sh
```

### Use more powerful instances
```bash
TASK_CPU=8192 TASK_MEMORY=16384 ./deploy.sh
```

### Deploy to different region
```bash
AWS_REGION=us-west-2 ./deploy.sh
```

## 🔍 Monitoring

### View logs
```bash
aws logs tail /ecs/litellm-benchmark-litellm --follow
```

### Check service status
```bash
aws ecs describe-services \
  --cluster litellm-benchmark-LiteLLM-Cluster \
  --services litellm-benchmark-litellm-service
```

### CloudWatch Metrics
- Go to AWS Console → CloudWatch → Metrics
- View ECS, RDS, and ALB metrics

## 🧹 Cleanup

When you're done testing:

```bash
aws cloudformation delete-stack --stack-name litellm-benchmark
```

This removes all resources and stops charges.

## ✨ Key Features

- ✅ **Validated Template** - CloudFormation template passed AWS validation
- ✅ **Production Ready** - High availability across multiple AZs
- ✅ **Secure by Default** - Private subnets, security groups, encrypted secrets
- ✅ **Cost Optimized** - Right-sized for performance and budget
- ✅ **Auto-scaling Ready** - Easy to configure auto-scaling
- ✅ **Well Documented** - Comprehensive guides included
- ✅ **Tested** - Includes validation and load testing tools

## 📊 Performance Benchmarks

Based on the [official benchmark guide](https://docs.litellm.ai/docs/benchmarks):

| Configuration | Median Latency | Throughput | LiteLLM Overhead |
|---------------|----------------|------------|------------------|
| 2 instances   | 200 ms         | 1,035 RPS  | 12 ms            |
| **4 instances** | **100 ms**   | **1,170 RPS** | **2 ms**     |

**Key Insight:** Doubling from 2 to 4 instances halves median latency!

## 🆘 Troubleshooting

### Tasks not starting?
- Check ECS service events in AWS Console
- View logs: `aws logs tail /ecs/litellm-benchmark-litellm --follow`

### Health checks failing?
- Wait 2-3 minutes for tasks to fully start
- Verify security groups allow ALB → ECS communication

### High latency?
- Ensure all 4 tasks are running
- Check CloudWatch metrics for CPU/Memory usage
- Run `./test-deployment.sh` to diagnose

### API authentication errors?
- Verify your master key is correct
- Check Secrets Manager for stored credentials

## 📚 Additional Resources

- **LiteLLM Documentation:** https://docs.litellm.ai
- **Benchmark Guide:** https://docs.litellm.ai/docs/benchmarks
- **GitHub Repository:** https://github.com/BerriAI/litellm
- **AWS ECS Best Practices:** https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/

## 🤝 Support

- **Issues:** https://github.com/BerriAI/litellm/issues
- **Discussions:** https://github.com/BerriAI/litellm/discussions

## 📝 Checklist

Before deploying:
- [ ] AWS CLI installed and configured
- [ ] AWS credentials with appropriate permissions
- [ ] Strong database password ready (min 8 chars)
- [ ] Strong master key ready (min 16 chars)
- [ ] Selected AWS region
- [ ] Reviewed cost estimates

After deploying:
- [ ] Saved API endpoint
- [ ] Saved master key securely
- [ ] Verified all tasks running
- [ ] Made test API call
- [ ] Ran validation script
- [ ] Ran benchmark test

---

## 🎉 Ready to Deploy?

```bash
./deploy.sh
```

**Deployment time:** ~10-15 minutes
**Expected performance:** ~100ms median latency, ~1,170 RPS
**Cost:** ~$440-460/month

---

**Created:** February 2026
**Benchmark Reference:** https://docs.litellm.ai/docs/benchmarks
**Template Status:** ✅ Validated with AWS CloudFormation
