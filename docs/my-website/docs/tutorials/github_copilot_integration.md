import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GitHub Copilot + LiteLLM Integration Tutorial

This comprehensive tutorial shows how to use GitHub Copilot with LiteLLM Proxy, enabling you to access multiple LLM providers through GitHub Copilot's interface with centralized management, cost tracking, and advanced features.

## Benefits of Using GitHub Copilot with LiteLLM

When you integrate GitHub Copilot with LiteLLM Proxy, you get:

- **Universal Model Access**: Access 100+ LLM providers (OpenAI, Anthropic, Google, etc.) through GitHub Copilot's familiar interface
- **Cost Management**: Track spending and set budgets across all GitHub Copilot usage
- **Centralized Control**: Manage access to multiple models through a single LiteLLM Proxy
- **Enhanced Security**: Implement authentication, rate limiting, and access controls
- **Load Balancing**: Distribute requests across multiple model deployments
- **Fallback Logic**: Automatic failover to backup models when primary models are unavailable

## Prerequisites

- GitHub Copilot subscription (Individual, Business, or Enterprise)
- Python 3.8+ installed
- Basic familiarity with REST APIs and authentication

## Quick Start

### 1. Install LiteLLM

```bash
pip install litellm[proxy]
```

### 2. Set Up Your Configuration

Create a `config.yaml` file with your model configurations:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: os.environ/GOOGLE_API_KEY

general_settings:
  master_key: your-secret-key # Change this to a secure key
  database_url: "sqlite:///litellm_proxy.db"
```

### 3. Start LiteLLM Proxy

```bash
litellm --config config.yaml --port 4000
```

Your LiteLLM proxy will be running at `http://localhost:4000`

### 4. Configure GitHub Copilot

GitHub Copilot can be configured to use custom model endpoints through various methods:

<Tabs>
<TabItem value="vscode" label="VS Code">

Install the GitHub Copilot extension and configure it to use your LiteLLM proxy:

1. Open VS Code Settings
2. Search for "github.copilot"
3. Add the following to your `settings.json`:

```json
{
  "github.copilot.advanced": {
    "debug.overrideProxyUrl": "http://localhost:4000",
    "debug.testOverrideProxyUrl": "http://localhost:4000"
  }
}
```

</TabItem>
<TabItem value="api" label="Direct API Usage">

You can also call GitHub Copilot models directly through the LiteLLM proxy:

```bash
curl -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user", 
        "content": "Write a Python function to calculate fibonacci numbers"
      }
    ]
  }'
```

</TabItem>
</Tabs>

## Advanced Configuration

### Authentication & Security

For production deployments, implement proper authentication:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  
  # Enable authentication
  ui_username: admin
  ui_password: os.environ/UI_PASSWORD
  
  # Database for persistence
  database_url: postgresql://user:password@localhost:5432/litellm_db
  
  # Rate limiting
  tpm_limit: 1000
  rpm_limit: 100
  max_budget: 100.0
```

### Load Balancing & Fallbacks

Configure multiple deployments with automatic fallback:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1
  
  - model_name: gpt-4o  # Same model name for load balancing
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2024-02-15-preview"

router_settings:
  routing_strategy: simple-shuffle  # Load balance requests
  model_group_alias:
    gpt-4-primary: gpt-4o
```

### Cost Tracking & Budgets

Monitor and control costs across all GitHub Copilot usage:

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://user:password@localhost:5432/litellm_db
  
  # Budget controls
  max_budget: 500.0  # $500/month limit
  budget_duration: "30d"
  
  # Cost tracking
  success_callback: ["langfuse", "posthog"]
  
litellm_settings:
  # Track costs per user/project
  track_cost_per_request: true
  store_model_in_db: true
```

## Production Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM ghcr.io/berriai/litellm:main-latest

COPY config.yaml /app/config.yaml

EXPOSE 4000

CMD ["--config", "/app/config.yaml", "--port", "4000", "--num_workers", "8"]
```

Build and run:

```bash
docker build -t my-litellm-proxy .
docker run -p 4000:4000 \
  -e OPENAI_API_KEY=your_openai_key \
  -e ANTHROPIC_API_KEY=your_anthropic_key \
  my-litellm-proxy
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm-proxy
  template:
    metadata:
      labels:
        app: litellm-proxy
    spec:
      containers:
      - name: litellm-proxy
        image: ghcr.io/berriai/litellm:main-latest
        ports:
        - containerPort: 4000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        command: ["litellm"]
        args: ["--config", "/app/config.yaml", "--port", "4000"]
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
      volumes:
      - name: config
        configMap:
          name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-proxy-service
spec:
  selector:
    app: litellm-proxy
  ports:
  - port: 4000
    targetPort: 4000
  type: LoadBalancer
```

## Usage Examples

### Code Completion

With GitHub Copilot configured to use your LiteLLM proxy, you'll get code suggestions powered by your chosen models:

```python
# Type this comment and Copilot will suggest implementations
# Function to process user data with validation
```

### Chat Interface

Use GitHub Copilot Chat with multiple model backends:

```
@copilot explain this code using Claude's reasoning capabilities
```

### API Integration

Call the proxy directly for custom integrations:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="your-secret-key"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet",
    messages=[
        {"role": "user", "content": "Explain async/await in Python"}
    ]
)

print(response.choices[0].message.content)
```

## Monitoring & Observability

### Built-in Dashboard

Access the LiteLLM admin dashboard at `http://localhost:4000/ui` to monitor:

- Request volume and latency
- Cost tracking per model
- Error rates and debugging
- User activity and quotas

### Integration with External Tools

Configure observability integrations:

```yaml
general_settings:
  success_callback: ["langfuse", "wandb", "lunary"]
  failure_callback: ["sentry"]

litellm_settings:
  set_verbose: true
  json_logs: true
```

## Troubleshooting

### Common Issues

**GitHub Copilot not using the proxy:**
- Verify the proxy URL is correctly configured in VS Code settings
- Check that the LiteLLM proxy is running and accessible
- Ensure the master key is properly set

**Authentication errors:**
- Verify your API keys are correctly set in environment variables
- Check that the master key matches between client and server
- Ensure the model names in your config match what you're requesting

**Rate limiting issues:**
- Review your rate limits in the configuration
- Check the LiteLLM logs for rate limit errors
- Consider implementing proper user authentication for better quota management

### Debug Mode

Enable debug logging:

```bash
export LITELLM_LOG=DEBUG
litellm --config config.yaml --debug
```

### Health Checks

Verify your proxy is working:

```bash
curl http://localhost:4000/health
```

Check model availability:

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## Best Practices

1. **Security First**: Always use environment variables for API keys and use strong master keys
2. **Monitor Costs**: Set up budget alerts and track usage per user/project
3. **High Availability**: Deploy with multiple replicas and implement health checks
4. **Rate Limiting**: Configure appropriate limits to prevent abuse
5. **Logging**: Enable comprehensive logging for debugging and compliance
6. **Testing**: Test failover scenarios and model switching

## Next Steps

- Explore [LiteLLM Router Documentation](../routing) for advanced load balancing
- Set up [observability integrations](../observability/langfuse_integration) for detailed analytics
- Learn about [enterprise features](../enterprise) for larger organizations
- Check out [deployment guides](../simple_proxy) for production environments

## Support

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [GitHub Issues](https://github.com/BerriAI/litellm/issues)
- [Discord Community](https://discord.gg/wuPM9dRgDw) 