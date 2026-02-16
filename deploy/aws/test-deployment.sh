#!/bin/bash
set -e

# LiteLLM AWS Deployment Test Script
# This script validates your AWS deployment and checks if it meets benchmark specifications

echo "=========================================="
echo "LiteLLM Deployment Validation"
echo "=========================================="
echo ""

# Configuration
STACK_NAME="${STACK_NAME:-litellm-benchmark}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Check if stack exists
echo "Checking if CloudFormation stack exists..."
if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" &> /dev/null; then
    echo "Error: Stack '$STACK_NAME' not found in region '$AWS_REGION'"
    echo "Have you deployed yet? Run ./deploy.sh first."
    exit 1
fi

echo "✓ Stack found"
echo ""

# Get stack outputs
echo "Retrieving deployment information..."
LOAD_BALANCER_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
  --output text)

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
  --output text)

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ECSServiceName`].OutputValue' \
  --output text)

echo "✓ Deployment information retrieved"
echo ""

# Test 1: Check ECS Service
echo "Test 1: ECS Service Status"
echo "----------------------------"
SERVICE_STATUS=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --region "$AWS_REGION" \
  --query 'services[0].[runningCount,desiredCount]' \
  --output text)

RUNNING_COUNT=$(echo $SERVICE_STATUS | awk '{print $1}')
DESIRED_COUNT=$(echo $SERVICE_STATUS | awk '{print $2}')

echo "Running tasks: $RUNNING_COUNT"
echo "Desired tasks: $DESIRED_COUNT"

if [ "$RUNNING_COUNT" -eq "$DESIRED_COUNT" ] && [ "$RUNNING_COUNT" -ge 4 ]; then
    echo "✓ All tasks are running"
else
    echo "⚠ Not all tasks are running yet"
    echo "  Expected: 4 or more tasks"
    echo "  Running: $RUNNING_COUNT"
fi
echo ""

# Test 2: Check Task Configuration
echo "Test 2: Task Configuration"
echo "----------------------------"
TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --region "$AWS_REGION" \
  --query 'services[0].taskDefinition' \
  --output text)

TASK_CONFIG=$(aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --region "$AWS_REGION" \
  --query 'taskDefinition.[cpu,memory]' \
  --output text)

TASK_CPU=$(echo $TASK_CONFIG | awk '{print $1}')
TASK_MEMORY=$(echo $TASK_CONFIG | awk '{print $2}')

echo "CPU per task: $TASK_CPU units ($(($TASK_CPU / 1024)) vCPU)"
echo "Memory per task: $TASK_MEMORY MB"

if [ "$TASK_CPU" -ge 4096 ] && [ "$TASK_MEMORY" -ge 8192 ]; then
    echo "✓ Task resources match benchmark configuration"
else
    echo "⚠ Task resources are lower than benchmark specification"
    echo "  Benchmark: 4096 CPU (4 vCPU), 8192 MB (8 GB)"
fi
echo ""

# Test 3: Health Check
echo "Test 3: Health Endpoint"
echo "------------------------"
echo "Testing $LOAD_BALANCER_URL/health/readiness"

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$LOAD_BALANCER_URL/health/readiness" 2>&1 || echo "000")

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✓ Health check passed (HTTP $HEALTH_RESPONSE)"
else
    echo "✗ Health check failed (HTTP $HEALTH_RESPONSE)"
    echo "  The service may still be starting up."
    echo "  Wait a few minutes and try again."
fi
echo ""

# Test 4: API Response Time
echo "Test 4: API Response Time"
echo "--------------------------"

# Get master key from Secrets Manager
MASTER_KEY=$(aws secretsmanager get-secret-value \
  --secret-id "$STACK_NAME-master-key" \
  --region "$AWS_REGION" \
  --query SecretString \
  --output text 2>/dev/null || echo "")

if [ -z "$MASTER_KEY" ]; then
    echo "⚠ Could not retrieve master key from Secrets Manager"
    echo "  Please provide the master key manually to test the API."
    echo ""
else
    echo "Testing API endpoint..."

    # Make 5 test requests and measure response time
    TOTAL_TIME=0
    SUCCESS_COUNT=0

    for i in {1..5}; do
        START_TIME=$(date +%s%3N)
        RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
          -X POST "$LOAD_BALANCER_URL/v1/chat/completions" \
          -H "Authorization: Bearer $MASTER_KEY" \
          -H "Content-Type: application/json" \
          -d '{"model":"fake-openai-endpoint","messages":[{"role":"user","content":"test"}]}' \
          2>&1 || echo "000")
        END_TIME=$(date +%s%3N)

        RESPONSE_TIME=$((END_TIME - START_TIME))

        if [ "$RESPONSE" = "200" ]; then
            echo "  Request $i: ${RESPONSE_TIME}ms (HTTP $RESPONSE)"
            TOTAL_TIME=$((TOTAL_TIME + RESPONSE_TIME))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  Request $i: Failed (HTTP $RESPONSE)"
        fi
    done

    if [ $SUCCESS_COUNT -gt 0 ]; then
        AVG_TIME=$((TOTAL_TIME / SUCCESS_COUNT))
        echo ""
        echo "Average response time: ${AVG_TIME}ms"

        if [ $AVG_TIME -le 200 ]; then
            echo "✓ Response time is good (target: ~100-200ms under load)"
        else
            echo "⚠ Response time is higher than expected"
            echo "  Note: Single requests may be slower. Run load tests for accurate results."
        fi
    else
        echo "✗ All API requests failed"
    fi
fi
echo ""

# Test 5: Database Connection
echo "Test 5: Database Connection"
echo "----------------------------"
DB_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
  --output text)

DB_STATUS=$(aws rds describe-db-instances \
  --region "$AWS_REGION" \
  --query "DBInstances[?Endpoint.Address=='$DB_ENDPOINT'].DBInstanceStatus" \
  --output text 2>/dev/null || echo "unknown")

echo "Database endpoint: $DB_ENDPOINT"
echo "Database status: $DB_STATUS"

if [ "$DB_STATUS" = "available" ]; then
    echo "✓ Database is available"
else
    echo "⚠ Database status: $DB_STATUS"
fi
echo ""

# Test 6: Load Balancer Health
echo "Test 6: Load Balancer Targets"
echo "-------------------------------"
TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups \
  --region "$AWS_REGION" \
  --query "TargetGroups[?contains(TargetGroupName, '$STACK_NAME')].TargetGroupArn" \
  --output text 2>/dev/null || echo "")

if [ -n "$TARGET_GROUP_ARN" ]; then
    HEALTHY_TARGETS=$(aws elbv2 describe-target-health \
      --target-group-arn "$TARGET_GROUP_ARN" \
      --region "$AWS_REGION" \
      --query "TargetHealthDescriptions[?TargetHealth.State=='healthy'] | length(@)" \
      --output text)

    TOTAL_TARGETS=$(aws elbv2 describe-target-health \
      --target-group-arn "$TARGET_GROUP_ARN" \
      --region "$AWS_REGION" \
      --query "length(TargetHealthDescriptions)" \
      --output text)

    echo "Healthy targets: $HEALTHY_TARGETS / $TOTAL_TARGETS"

    if [ "$HEALTHY_TARGETS" -ge 4 ]; then
        echo "✓ All targets are healthy"
    else
        echo "⚠ Not all targets are healthy yet"
    fi
else
    echo "⚠ Could not find target group"
fi
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo "Deployment URL: $LOAD_BALANCER_URL"
echo "API Endpoint: $LOAD_BALANCER_URL/v1"
echo ""
echo "Benchmark Configuration:"
echo "  - Tasks: $RUNNING_COUNT / $DESIRED_COUNT"
echo "  - CPU per task: $TASK_CPU units"
echo "  - Memory per task: $TASK_MEMORY MB"
echo ""

# Recommendations
echo "Next Steps:"
echo ""
echo "1. Run a full benchmark test:"
echo "   pip install locust"
echo "   export LITELLM_MASTER_KEY='$MASTER_KEY'"
echo "   locust -f locustfile.py --host=$LOAD_BALANCER_URL \\"
echo "     --users=1000 --spawn-rate=500 --run-time=5m --headless"
echo ""
echo "2. Monitor your deployment:"
echo "   aws logs tail /ecs/$STACK_NAME-litellm --follow --region $AWS_REGION"
echo ""
echo "3. View CloudWatch metrics:"
echo "   https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION"
echo ""
echo "4. Compare results with benchmark:"
echo "   https://docs.litellm.ai/docs/benchmarks"
echo ""
