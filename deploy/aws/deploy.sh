#!/bin/bash
set -e

# LiteLLM Benchmark AWS ECS Deployment Script
# This script deploys LiteLLM on AWS ECS with the benchmark configuration:
# - 4 instances with 4 vCPUs and 8 GB RAM each
# - 4 workers per instance
# - PostgreSQL database
# - Application Load Balancer

echo "=========================================="
echo "LiteLLM Benchmark AWS ECS Deployment"
echo "=========================================="
echo ""

# Default values
STACK_NAME="${STACK_NAME:-litellm-benchmark}"
AWS_REGION="${AWS_REGION:-us-east-1}"
DESIRED_TASKS="${DESIRED_TASKS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TASK_CPU="${TASK_CPU:-4096}"
TASK_MEMORY="${TASK_MEMORY:-8192}"

# Prompt for required parameters
if [ -z "$DB_PASSWORD" ]; then
    echo "Enter PostgreSQL database password (min 8 characters):"
    read -s DB_PASSWORD
    echo ""
fi

if [ -z "$MASTER_KEY" ]; then
    echo "Enter LiteLLM Proxy Master Key (min 16 characters):"
    read -s MASTER_KEY
    echo ""
fi

# Validate inputs
if [ ${#DB_PASSWORD} -lt 8 ]; then
    echo "Error: Database password must be at least 8 characters long"
    exit 1
fi

if [ ${#MASTER_KEY} -lt 16 ]; then
    echo "Error: Master key must be at least 16 characters long"
    exit 1
fi

echo ""
echo "Deployment Configuration:"
echo "  Stack Name: $STACK_NAME"
echo "  AWS Region: $AWS_REGION"
echo "  Desired Tasks: $DESIRED_TASKS"
echo "  Workers per Task: $NUM_WORKERS"
echo "  CPU per Task: $TASK_CPU ($(($TASK_CPU / 1024)) vCPU)"
echo "  Memory per Task: $TASK_MEMORY MB"
echo "  Total Workers: $(($DESIRED_TASKS * $NUM_WORKERS))"
echo ""

# Confirm deployment
echo "This will create AWS resources that incur costs (~$440-460/month)."
echo "Do you want to proceed? (yes/no)"
read CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Starting deployment..."
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    echo "Visit: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials are not configured."
    echo "Run 'aws configure' to set up your credentials."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMPLATE_FILE="$SCRIPT_DIR/cloudformation-ecs.yaml"

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: CloudFormation template not found at $TEMPLATE_FILE"
    exit 1
fi

# Create the CloudFormation stack
echo "Creating CloudFormation stack..."
aws cloudformation create-stack \
  --stack-name "$STACK_NAME" \
  --template-body "file://$TEMPLATE_FILE" \
  --parameters \
    ParameterKey=DBPassword,ParameterValue="$DB_PASSWORD" \
    ParameterKey=MasterKey,ParameterValue="$MASTER_KEY" \
    ParameterKey=DesiredTaskCount,ParameterValue="$DESIRED_TASKS" \
    ParameterKey=NumWorkersPerTask,ParameterValue="$NUM_WORKERS" \
    ParameterKey=TaskCPU,ParameterValue="$TASK_CPU" \
    ParameterKey=TaskMemory,ParameterValue="$TASK_MEMORY" \
  --capabilities CAPABILITY_IAM \
  --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create CloudFormation stack"
    exit 1
fi

echo ""
echo "CloudFormation stack creation initiated."
echo "This will take approximately 10-15 minutes..."
echo ""
echo "You can monitor progress in the AWS Console:"
echo "https://console.aws.amazon.com/cloudformation/home?region=$AWS_REGION#/stacks"
echo ""
echo "Waiting for stack creation to complete..."

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Stack creation failed or timed out."
    echo "Check the CloudFormation console for details:"
    echo "https://console.aws.amazon.com/cloudformation/home?region=$AWS_REGION#/stacks"
    exit 1
fi

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo ""

# Get stack outputs
LOAD_BALANCER_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
  --output text)

API_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`APIEndpoint`].OutputValue' \
  --output text)

DATABASE_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
  --output text)

BENCHMARK_CONFIG=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`BenchmarkConfiguration`].OutputValue' \
  --output text)

echo "Deployment Details:"
echo "  Load Balancer URL: $LOAD_BALANCER_URL"
echo "  API Endpoint: $API_ENDPOINT"
echo "  Database Endpoint: $DATABASE_ENDPOINT"
echo "  Configuration: $BENCHMARK_CONFIG"
echo ""

echo "Master Key (save this securely):"
echo "  $MASTER_KEY"
echo ""

echo "Testing the deployment..."
echo ""

# Wait a bit for the service to be fully ready
sleep 10

# Test health endpoint
echo "1. Health check..."
if curl -s -f "$LOAD_BALANCER_URL/health/readiness" > /dev/null 2>&1; then
    echo "   ✓ Health check passed"
else
    echo "   ⚠ Health check not ready yet (this is normal, ECS tasks may still be starting)"
fi

echo ""
echo "Next Steps:"
echo ""
echo "1. Test the API:"
echo "   curl -X POST \"$API_ENDPOINT/chat/completions\" \\"
echo "     -H \"Authorization: Bearer $MASTER_KEY\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model\":\"fake-openai-endpoint\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "2. View logs:"
echo "   aws logs tail /ecs/$STACK_NAME-litellm --follow --region $AWS_REGION"
echo ""
echo "3. Run benchmark tests:"
echo "   See README.md for Locust load testing instructions"
echo ""
echo "4. To delete all resources when done:"
echo "   aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION"
echo ""
echo "Documentation: https://docs.litellm.ai/docs/benchmarks"
echo ""
