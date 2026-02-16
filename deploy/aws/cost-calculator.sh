#!/bin/bash

# AWS Cost Calculator for LiteLLM Deployment
# Estimates monthly costs based on AWS pricing (us-east-1)

echo "=========================================="
echo "LiteLLM AWS Deployment Cost Calculator"
echo "=========================================="
echo ""

# Get user inputs or use defaults
read -p "Number of ECS tasks (default: 4): " TASK_COUNT
TASK_COUNT=${TASK_COUNT:-4}

read -p "vCPU per task (default: 4): " VCPU_PER_TASK
VCPU_PER_TASK=${VCPU_PER_TASK:-4}

read -p "Memory per task in GB (default: 8): " MEMORY_PER_TASK
MEMORY_PER_TASK=${MEMORY_PER_TASK:-8}

read -p "RDS instance class (t3.micro/t3.small/t3.medium/r6g.large, default: t3.medium): " RDS_CLASS
RDS_CLASS=${RDS_CLASS:-t3.medium}

read -p "Estimated monthly data transfer in GB (default: 100): " DATA_TRANSFER
DATA_TRANSFER=${DATA_TRANSFER:-100}

echo ""
echo "=========================================="
echo "Cost Breakdown (Monthly, USD)"
echo "=========================================="
echo ""

# ECS Fargate Costs
# Pricing: $0.04048 per vCPU-hour, $0.004445 per GB-hour (us-east-1)
HOURS_PER_MONTH=730
FARGATE_CPU_COST_PER_HOUR=0.04048
FARGATE_MEMORY_COST_PER_HOUR=0.004445

TOTAL_VCPU=$(echo "$TASK_COUNT * $VCPU_PER_TASK" | bc)
TOTAL_MEMORY=$(echo "$TASK_COUNT * $MEMORY_PER_TASK" | bc)

CPU_COST=$(echo "$TOTAL_VCPU * $FARGATE_CPU_COST_PER_HOUR * $HOURS_PER_MONTH" | bc)
MEMORY_COST=$(echo "$TOTAL_MEMORY * $FARGATE_MEMORY_COST_PER_HOUR * $HOURS_PER_MONTH" | bc)
FARGATE_TOTAL=$(echo "$CPU_COST + $MEMORY_COST" | bc)

printf "ECS Fargate:\n"
printf "  Tasks: %d\n" $TASK_COUNT
printf "  vCPU: %d total (%d per task)\n" $TOTAL_VCPU $VCPU_PER_TASK
printf "  Memory: %d GB total (%d GB per task)\n" $TOTAL_MEMORY $MEMORY_PER_TASK
printf "  CPU cost: \$%.2f\n" $CPU_COST
printf "  Memory cost: \$%.2f\n" $MEMORY_COST
printf "  Subtotal: \$%.2f\n" $FARGATE_TOTAL
echo ""

# RDS Costs
case $RDS_CLASS in
  "t3.micro")
    RDS_COST=13.87
    ;;
  "t3.small")
    RDS_COST=27.74
    ;;
  "t3.medium")
    RDS_COST=55.48
    ;;
  "r6g.large")
    RDS_COST=153.00
    ;;
  *)
    RDS_COST=55.48
    RDS_CLASS="t3.medium"
    ;;
esac

# Add storage cost (100 GB GP3)
STORAGE_COST=11.50
RDS_TOTAL=$(echo "$RDS_COST + $STORAGE_COST" | bc)

printf "RDS PostgreSQL:\n"
printf "  Instance class: db.%s\n" $RDS_CLASS
printf "  Storage: 100 GB GP3\n"
printf "  Instance cost: \$%.2f\n" $RDS_COST
printf "  Storage cost: \$%.2f\n" $STORAGE_COST
printf "  Subtotal: \$%.2f\n" $RDS_TOTAL
echo ""

# Application Load Balancer
ALB_COST=18.40  # ~$0.025/hour = $18.40/month
ALB_LCU_COST=5.60  # Estimated LCU cost

ALB_TOTAL=$(echo "$ALB_COST + $ALB_LCU_COST" | bc)

printf "Application Load Balancer:\n"
printf "  Fixed cost: \$%.2f\n" $ALB_COST
printf "  LCU cost (estimated): \$%.2f\n" $ALB_LCU_COST
printf "  Subtotal: \$%.2f\n" $ALB_TOTAL
echo ""

# NAT Gateway
NAT_COST=32.85  # $0.045/hour = $32.85/month
NAT_DATA_COST=$(echo "$DATA_TRANSFER * 0.045" | bc)

NAT_TOTAL=$(echo "$NAT_COST + $NAT_DATA_COST" | bc)

printf "NAT Gateway:\n"
printf "  Fixed cost: \$%.2f\n" $NAT_COST
printf "  Data processing (%d GB): \$%.2f\n" $DATA_TRANSFER $NAT_DATA_COST
printf "  Subtotal: \$%.2f\n" $NAT_TOTAL
echo ""

# Data Transfer Out
DATA_TRANSFER_COST=$(echo "$DATA_TRANSFER * 0.09" | bc)

printf "Data Transfer:\n"
printf "  Outbound data (%d GB): \$%.2f\n" $DATA_TRANSFER $DATA_TRANSFER_COST
printf "  Subtotal: \$%.2f\n" $DATA_TRANSFER_COST
echo ""

# Secrets Manager
SECRETS_COST=0.80  # 2 secrets × $0.40/secret/month

printf "Secrets Manager:\n"
printf "  2 secrets: \$%.2f\n" $SECRETS_COST
echo ""

# CloudWatch Logs
LOGS_INGESTION=5.00  # Estimated based on volume
LOGS_STORAGE=3.00    # Estimated for 7 days retention

LOGS_TOTAL=$(echo "$LOGS_INGESTION + $LOGS_STORAGE" | bc)

printf "CloudWatch Logs:\n"
printf "  Ingestion: \$%.2f\n" $LOGS_INGESTION
printf "  Storage: \$%.2f\n" $LOGS_STORAGE
printf "  Subtotal: \$%.2f\n" $LOGS_TOTAL
echo ""

# Total
TOTAL=$(echo "$FARGATE_TOTAL + $RDS_TOTAL + $ALB_TOTAL + $NAT_TOTAL + $DATA_TRANSFER_COST + $SECRETS_COST + $LOGS_TOTAL" | bc)

echo "=========================================="
printf "TOTAL MONTHLY COST: \$%.2f\n" $TOTAL
echo "=========================================="
echo ""

# Savings with Reserved Capacity
echo "Potential Savings with Reserved Capacity:"
echo "------------------------------------------"

FARGATE_SAVINGS_1Y=$(echo "$FARGATE_TOTAL * 0.25" | bc)
FARGATE_SAVINGS_3Y=$(echo "$FARGATE_TOTAL * 0.45" | bc)

RDS_SAVINGS_1Y=$(echo "$RDS_COST * 0.30" | bc)
RDS_SAVINGS_3Y=$(echo "$RDS_COST * 0.60" | bc)

TOTAL_SAVINGS_1Y=$(echo "$FARGATE_SAVINGS_1Y + $RDS_SAVINGS_1Y" | bc)
TOTAL_SAVINGS_3Y=$(echo "$FARGATE_SAVINGS_3Y + $RDS_SAVINGS_3Y" | bc)

TOTAL_WITH_1Y=$(echo "$TOTAL - $TOTAL_SAVINGS_1Y" | bc)
TOTAL_WITH_3Y=$(echo "$TOTAL - $TOTAL_SAVINGS_3Y" | bc)

printf "1-Year Reserved:\n"
printf "  Fargate Savings Plan: -\$%.2f (25%%)\n" $FARGATE_SAVINGS_1Y
printf "  RDS Reserved Instance: -\$%.2f (30%%)\n" $RDS_SAVINGS_1Y
printf "  New Total: \$%.2f (saves \$%.2f/month)\n" $TOTAL_WITH_1Y $TOTAL_SAVINGS_1Y
echo ""

printf "3-Year Reserved:\n"
printf "  Fargate Savings Plan: -\$%.2f (45%%)\n" $FARGATE_SAVINGS_3Y
printf "  RDS Reserved Instance: -\$%.2f (60%%)\n" $RDS_SAVINGS_3Y
printf "  New Total: \$%.2f (saves \$%.2f/month)\n" $TOTAL_WITH_3Y $TOTAL_SAVINGS_3Y
echo ""

# Annual costs
ANNUAL=$(echo "$TOTAL * 12" | bc)
ANNUAL_1Y=$(echo "$TOTAL_WITH_1Y * 12" | bc)
ANNUAL_3Y=$(echo "$TOTAL_WITH_3Y * 12" | bc)

echo "Annual Costs:"
echo "-------------"
printf "Pay-as-you-go: \$%.2f/year\n" $ANNUAL
printf "1-Year Reserved: \$%.2f/year (saves \$%.2f)\n" $ANNUAL_1Y $(echo "$ANNUAL - $ANNUAL_1Y" | bc)
printf "3-Year Reserved: \$%.2f/year (saves \$%.2f)\n" $ANNUAL_3Y $(echo "$ANNUAL - $ANNUAL_3Y" | bc)
echo ""

# Alternative configurations
echo "=========================================="
echo "Alternative Configurations"
echo "=========================================="
echo ""

echo "Development/Testing (2 tasks, 2 vCPU, 4 GB, t3.micro):"
echo "  Estimated cost: ~\$150-180/month"
echo ""

echo "Production High-Availability (8 tasks, 4 vCPU, 8 GB, r6g.large Multi-AZ):"
echo "  Estimated cost: ~\$1,200-1,500/month"
echo ""

echo "Note: Costs are estimates based on us-east-1 pricing."
echo "Actual costs may vary based on usage patterns, region, and AWS pricing changes."
echo "Use AWS Cost Calculator for precise estimates: https://calculator.aws/"
echo ""
