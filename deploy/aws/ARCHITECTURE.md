# AWS Deployment Architecture

This document describes the architecture of the LiteLLM AWS deployment configured for benchmark performance.

## Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   Internet      │
                                    └────────┬────────┘
                                             │
                                             │ HTTPS/HTTP
                                             │
                   ┌─────────────────────────▼───────────────────────────┐
                   │         Application Load Balancer (ALB)             │
                   │                                                      │
                   │  - Internet-facing                                   │
                   │  - HTTP/HTTPS listeners                              │
                   │  - Health checks: /health/readiness                  │
                   └──────────────────┬───────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      │               │               │
        ┌─────────────▼──┐  ┌────────▼────┐  ┌──────▼───────────┐
        │  ECS Task 1    │  │ ECS Task 2  │  │  ECS Task 3-4    │
        │  (Fargate)     │  │ (Fargate)   │  │  (Fargate)       │
        │                │  │             │  │                  │
        │  - 4 vCPU      │  │ - 4 vCPU    │  │  - 4 vCPU        │
        │  - 8 GB RAM    │  │ - 8 GB RAM  │  │  - 8 GB RAM      │
        │  - 4 workers   │  │ - 4 workers │  │  - 4 workers     │
        │                │  │             │  │                  │
        │  LiteLLM       │  │  LiteLLM    │  │  LiteLLM         │
        │  Port: 4000    │  │  Port: 4000 │  │  Port: 4000      │
        └────────┬───────┘  └──────┬──────┘  └────────┬─────────┘
                 │                 │                   │
                 └─────────────────┼───────────────────┘
                                   │
                                   │ PostgreSQL Protocol
                                   │ Port: 5432
                                   │
                         ┌─────────▼──────────┐
                         │  RDS PostgreSQL    │
                         │                    │
                         │  - db.t3.medium    │
                         │  - 2 vCPU          │
                         │  - 4 GB RAM        │
                         │  - 100 GB Storage  │
                         │  - Multi-AZ        │
                         │  - Auto backup     │
                         └────────────────────┘
```

## Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          VPC (10.0.0.0/16)                      │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Public Subnets (2 AZs)                       │ │
│  │                                                            │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐     │ │
│  │  │  Public Subnet 1    │    │  Public Subnet 2    │     │ │
│  │  │  (10.0.1.0/24)      │    │  (10.0.2.0/24)      │     │ │
│  │  │                     │    │                     │     │ │
│  │  │  - ALB              │    │  - ALB              │     │ │
│  │  │  - NAT Gateway      │    │                     │     │ │
│  │  │  - Internet Gateway │    │                     │     │ │
│  │  └─────────────────────┘    └─────────────────────┘     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Private Subnets (2 AZs)                      │ │
│  │                                                            │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐     │ │
│  │  │  Private Subnet 1   │    │  Private Subnet 2   │     │ │
│  │  │  (10.0.11.0/24)     │    │  (10.0.12.0/24)     │     │ │
│  │  │                     │    │                     │     │ │
│  │  │  - ECS Tasks        │    │  - ECS Tasks        │     │ │
│  │  │  - RDS Primary      │    │  - RDS Standby      │     │ │
│  │  └─────────────────────┘    └─────────────────────┘     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Security Groups

```
┌──────────────────────────────────────────────────────────────┐
│                      Security Groups                          │
└──────────────────────────────────────────────────────────────┘

┌─────────────────┐       ┌─────────────────┐      ┌──────────────────┐
│  ALB Security   │       │  ECS Security   │      │  RDS Security    │
│     Group       │       │     Group       │      │     Group        │
│                 │       │                 │      │                  │
│  Inbound:       │       │  Inbound:       │      │  Inbound:        │
│  - 80 (HTTP)    │──────▶│  - 4000 (HTTP)  │─────▶│  - 5432 (PG)     │
│  - 443 (HTTPS)  │  from │    from ALB SG  │ from │    from ECS SG   │
│    from 0.0.0.0 │  ALB  │                 │ ECS  │                  │
│                 │       │  Outbound:      │      │  Outbound:       │
│  Outbound:      │       │  - All          │      │  - All           │
│  - All          │       │                 │      │                  │
└─────────────────┘       └─────────────────┘      └──────────────────┘
```

## Components

### 1. Application Load Balancer (ALB)

**Purpose:** Distributes incoming traffic across ECS tasks

**Configuration:**
- Type: Application Load Balancer
- Scheme: Internet-facing
- Subnets: Public subnets in 2 availability zones
- Listeners: HTTP (port 80), optionally HTTPS (port 443)
- Health check: `/health/readiness`
- Health check interval: 30 seconds
- Healthy threshold: 2 consecutive successes
- Unhealthy threshold: 3 consecutive failures

**Benefits:**
- Automatic SSL termination (with HTTPS)
- Health monitoring and automatic failover
- Connection draining during deployments
- Path-based routing (if needed)

### 2. ECS Fargate Tasks

**Purpose:** Run LiteLLM proxy containers

**Configuration:**
- Launch type: Fargate
- Task count: 4 (configurable)
- CPU: 4 vCPU (4096 units) per task
- Memory: 8 GB (8192 MB) per task
- Workers: 4 per task
- Total capacity: 16 vCPU, 32 GB RAM, 16 workers

**Container Configuration:**
- Image: `ghcr.io/berriai/litellm-database:main-latest`
- Port: 4000
- Command: `--port 4000 --num_workers 4`
- Health check: HTTP GET `/health/liveliness`
- Environment variables:
  - `DATABASE_URL`: PostgreSQL connection string
  - `STORE_MODEL_IN_DB`: True
  - `PROXY_MASTER_KEY`: From Secrets Manager

**Benefits:**
- Serverless containers (no EC2 management)
- Automatic scaling capability
- High availability across AZs
- Isolated execution environment

### 3. RDS PostgreSQL

**Purpose:** Persistent storage for LiteLLM configuration and logs

**Configuration:**
- Engine: PostgreSQL 16.3
- Instance class: db.t3.medium (2 vCPU, 4 GB RAM)
- Storage: 100 GB GP3 SSD
- Multi-AZ: No (can be enabled for HA)
- Backup retention: 7 days
- Automated backups: Yes

**Database Schema:**
- Managed by Prisma ORM
- Tables: models, users, teams, keys, logs, etc.
- Automatic migrations on deployment

**Recommended Settings:**
```sql
-- For 1-2K RPS workload
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
work_mem = 5MB
```

**Benefits:**
- Automatic backups and point-in-time recovery
- Automatic software patching
- Monitoring via CloudWatch
- Easy scaling (vertical and storage)

### 4. VPC and Networking

**Configuration:**
- VPC CIDR: 10.0.0.0/16
- Public Subnets: 10.0.1.0/24, 10.0.2.0/24
- Private Subnets: 10.0.11.0/24, 10.0.12.0/24
- NAT Gateway: 1 (in Public Subnet 1)
- Internet Gateway: 1

**Routing:**
- Public subnets → Internet Gateway
- Private subnets → NAT Gateway → Internet Gateway

**Benefits:**
- ECS tasks in private subnets for security
- Database isolated from internet
- Controlled outbound access via NAT Gateway
- High availability across 2 AZs

### 5. Secrets Management

**Configuration:**
- AWS Secrets Manager for sensitive data
- Secrets:
  - Database password
  - LiteLLM master key
  - API keys (stored separately)

**Benefits:**
- Encrypted at rest
- Automatic rotation support
- Audit logging via CloudTrail
- Fine-grained IAM access control

### 6. Logging and Monitoring

**CloudWatch Logs:**
- Log group: `/ecs/[stack-name]-litellm`
- Retention: 7 days (configurable)
- Logs from all ECS tasks

**CloudWatch Metrics:**
- ECS: CPU, Memory, Task Count
- ALB: Request Count, Latency, Target Health
- RDS: Connections, CPU, Storage

**Custom Metrics:**
- LiteLLM reports overhead in `x-litellm-overhead-duration-ms` header
- Can be extracted and sent to CloudWatch

## Data Flow

### Request Flow

1. **Client Request**
   ```
   Client → ALB (port 80/443)
   ```

2. **Load Balancing**
   ```
   ALB → Target Group → Healthy ECS Tasks
   ```
   - ALB selects a healthy task using round-robin
   - Sticky sessions not enabled (stateless)

3. **LiteLLM Processing**
   ```
   ECS Task → LiteLLM Proxy (4 workers)
   ```
   - Request handled by one of 4 workers
   - Worker selection by internal load balancing (Uvicorn)

4. **Database Operations**
   ```
   LiteLLM → RDS PostgreSQL
   ```
   - Validate API key
   - Log request
   - Retrieve model configuration

5. **External LLM Call**
   ```
   LiteLLM → External LLM Provider (OpenAI, Anthropic, etc.)
   ```
   - Transform request to provider format
   - Forward request via NAT Gateway
   - Receive and transform response

6. **Response Flow**
   ```
   LiteLLM → ALB → Client
   ```
   - Response sent back through ALB
   - Overhead metrics in headers

### Database Connection Pooling

```
┌──────────────────────────────────────────────┐
│  4 ECS Tasks × 4 Workers = 16 Workers       │
│                                              │
│  Each Worker → Connection Pool               │
│  Pool size: ~10 connections per worker       │
│  Total connections: ~160                     │
│                                              │
│  RDS max_connections: 200                    │
│  Available headroom: 40 connections          │
└──────────────────────────────────────────────┘
```

## High Availability

### Availability Zones

- Resources deployed across 2 AZs
- ECS tasks distributed automatically
- RDS can be configured for Multi-AZ
- ALB spans both AZs

### Failure Scenarios

**Single ECS Task Failure:**
- ALB marks task unhealthy
- Traffic routed to other tasks
- ECS starts replacement task
- Impact: 25% capacity reduction (temporary)

**Availability Zone Failure:**
- ALB routes all traffic to healthy AZ
- ECS maintains tasks in remaining AZ
- Impact: 50% capacity reduction (until AZ recovers)

**Database Failure:**
- With Multi-AZ: Automatic failover to standby (~60-120s)
- Without Multi-AZ: Manual restore from backup

### Recovery Time Objectives

| Scenario | RTO | RPO |
|----------|-----|-----|
| Single task failure | < 2 minutes | None (stateless) |
| AZ failure | < 1 minute | None (stateless) |
| Database failure (Multi-AZ) | < 2 minutes | ~0 (sync replication) |
| Database failure (Single-AZ) | 30-60 minutes | ~5 minutes (backup) |
| Complete region failure | Hours | Depends on backup strategy |

## Scaling

### Horizontal Scaling (Task Count)

**Manual Scaling:**
```bash
aws ecs update-service \
  --cluster [cluster-name] \
  --service [service-name] \
  --desired-count 8
```

**Auto Scaling (CPU-based):**
- Scale out: When average CPU > 70%
- Scale in: When average CPU < 30%
- Min tasks: 2
- Max tasks: 10

**Expected Performance by Scale:**

| Tasks | Workers | Expected RPS | Median Latency |
|-------|---------|--------------|----------------|
| 2     | 8       | ~1,035       | ~200ms         |
| 4     | 16      | ~1,170       | ~100ms         |
| 8     | 32      | ~2,000+      | ~50-75ms       |

### Vertical Scaling (Task Size)

**Upgrade to 8 vCPU, 16 GB:**
```yaml
TaskCPU: 8192
TaskMemory: 16384
```

**Benefits:**
- More workers per task (8-16 workers)
- Better performance per task
- Fewer tasks needed for same throughput

### Database Scaling

**Vertical Scaling:**
- Upgrade to db.r6g.large (2 vCPU → 8 vCPU)
- Minimal downtime (~1-2 minutes)

**Read Replicas:**
- Offload read queries
- Reduce primary load
- Not needed for typical LiteLLM workload

## Cost Optimization

### Reserved Capacity

**ECS Fargate Savings Plans:**
- 1-year: ~20-30% savings
- 3-year: ~40-50% savings
- Applies to Fargate compute usage

**RDS Reserved Instances:**
- 1-year: ~30% savings
- 3-year: ~60% savings
- Partial or full upfront payment

### Right-Sizing

**Monitor and adjust:**
- Use CloudWatch to track actual CPU/Memory usage
- Scale down if consistently under 50% utilization
- Scale up if consistently over 80% utilization

### Alternative Configurations

**Lower Cost (Dev/Test):**
- 2 tasks × 2 vCPU × 4 GB
- db.t3.micro
- Single AZ
- Cost: ~$150-200/month

**Production (HA + Performance):**
- 8 tasks × 4 vCPU × 8 GB
- db.r6g.large (Multi-AZ)
- Redis cluster
- Cost: ~$1,200-1,500/month

## Security Best Practices

### Network Security

- ✅ ECS tasks in private subnets
- ✅ Database not publicly accessible
- ✅ Security groups with principle of least privilege
- ✅ NAT Gateway for controlled outbound access
- ⚠️ Consider VPC endpoints for AWS services (S3, Secrets Manager)

### Authentication & Authorization

- ✅ Master key stored in Secrets Manager
- ✅ IAM roles for task execution
- ✅ IAM roles for task operations
- ⚠️ Implement key rotation policy
- ⚠️ Use IAM-based database authentication

### Data Protection

- ✅ RDS encryption at rest
- ✅ Secrets Manager encryption
- ✅ HTTPS termination at ALB (with certificate)
- ⚠️ Enable CloudTrail for audit logging
- ⚠️ Enable VPC Flow Logs

### Compliance

- Enable CloudWatch Logs encryption
- Configure S3 for long-term log archival
- Implement backup retention policies
- Regular security assessments

## Monitoring and Alerting

### Key Metrics to Monitor

**Application Performance:**
- Request latency (P50, P95, P99)
- Request rate (RPS)
- Error rate (4xx, 5xx)
- LiteLLM overhead (custom metric)

**Infrastructure Health:**
- ECS task count and health
- CPU/Memory utilization
- Database connections
- Target health

**Cost Metrics:**
- Fargate compute hours
- Data transfer costs
- RDS instance hours
- NAT Gateway data transfer

### Recommended Alarms

```yaml
Alarms:
  - High 5xx rate (> 1%)
  - High latency (P95 > 500ms)
  - Low healthy target count (< 2)
  - High database CPU (> 80%)
  - High database connections (> 180)
  - Task stopped unexpectedly
```

## References

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [AWS RDS Performance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)
- [LiteLLM Benchmark](https://docs.litellm.ai/docs/benchmarks)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
