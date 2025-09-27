````markdown
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Deploy LiteLLM Proxy on AWS EKS

Deploy LiteLLM Proxy on Amazon Elastic Kubernetes Service (EKS) for scalable LLM API management.

## Overview

This guide shows you how to deploy LiteLLM Proxy on AWS EKS in under 30 minutes. LiteLLM provides a unified API for multiple LLM providers (OpenAI, Anthropic, Azure, etc.) with features like load balancing, rate limiting, and cost tracking.

## Quick Start (5 minutes)

### Prerequisites

- AWS CLI configured with appropriate permissions
- `kubectl` installed
- `eksctl` installed (recommended)

### Step 1: Create EKS Cluster

```bash
# Create a basic EKS cluster
eksctl create cluster \
  --name litellm-cluster \
  --region us-east-1 \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

### Step 2: Deploy LiteLLM

Create a file called `litellm-deployment.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: litellm
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-config
  namespace: litellm
data:
  config.yaml: |
    model_list:
      - model_name: gpt-4o
        litellm_params:
          model: openai/gpt-4o
          api_key: os.environ/OPENAI_API_KEY
      - model_name: claude-3-5-sonnet
        litellm_params:
          model: anthropic/claude-3-5-sonnet-20241022
          api_key: os.environ/ANTHROPIC_API_KEY

    general_settings:
      master_key: os.environ/LITELLM_MASTER_KEY
---
apiVersion: v1
kind: Secret
metadata:
  name: litellm-secrets
  namespace: litellm
type: Opaque
data:
  OPENAI_API_KEY: <your-openai-key>
  ANTHROPIC_API_KEY: <your-anthropic-key>
  LITELLM_MASTER_KEY: <your-master-key>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: litellm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
    spec:
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm:main-stable
        ports:
        - containerPort: 4000
        envFrom:
        - secretRef:
            name: litellm-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /health/liveliness
            port: 4000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: config
        configMap:
          name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-service
  namespace: litellm
spec:
  selector:
    app: litellm
  ports:
    - port: 80
      targetPort: 4000
  type: LoadBalancer
```

Apply the deployment:

```bash
kubectl apply -f litellm-deployment.yaml
```

### Step 4: Get External IP and Test

```bash
# Get the LoadBalancer URL (may take a few minutes to provision)
kubectl get svc litellm-service -n litellm

# Test the API (replace YOUR_URL with the actual LoadBalancer URL)
curl -X POST http://<proxy-base-url>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello from EKS!"}]
  }'
```

🎉 **Congratulations!** You now have LiteLLM running on EKS!

## Production Deployment

For production use, enhance your deployment with these components:

### 1. Database Integration

Add PostgreSQL for persistent storage:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: litellm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: litellm
        - name: POSTGRES_USER
          value: litellm
        - name: POSTGRES_PASSWORD
          value: your-secure-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: litellm
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: litellm
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
```

Update your ConfigMap to use the database:

```yaml
data:
  config.yaml: |
    model_list:
      # ... your models ...
    
    general_settings:
      master_key: os.environ/LITELLM_MASTER_KEY
      database_url: postgresql://litellm:your-secure-password@postgres-service:5432/litellm
```

### 2. Auto Scaling

Add horizontal pod autoscaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-hpa
  namespace: litellm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. HTTPS Ingress

Set up SSL termination with AWS Load Balancer Controller:

```bash
# Install AWS Load Balancer Controller
eksctl create iamserviceaccount \
  --cluster=litellm-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
  --approve

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=litellm-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

Create an Ingress with SSL:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: litellm-ingress
  namespace: litellm
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456789012:certificate/your-cert-arn
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: litellm-service
            port:
              number: 80
```

## Configuration Examples

### Multiple LLM Providers

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-02-01"

  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: os.environ/GEMINI_API_KEY
```

### Rate Limiting and Routing

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 1000
      tpm: 100000

router_settings:
  routing_strategy: usage-based-routing
  fallbacks:
    - gpt-4o: ["claude-3-5-sonnet"]

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

## Using Helm Chart

For easier management, use the official Helm chart:

```bash
# Add the Helm repository
helm repo add litellm https://berriai.github.io/litellm-helm
helm repo update

# Install with basic configuration
helm install litellm litellm/litellm \
  --namespace litellm \
  --create-namespace \
  --set masterkey=your-master-key \
  --set environmentSecrets[0]=litellm-secrets
```

## Monitoring

### Health Checks

LiteLLM provides built-in health endpoints:

```bash
# Check liveliness
curl http://your-service/health/liveliness

# Check readiness
curl http://your-service/health/readiness
```

### Prometheus Metrics

Enable Prometheus metrics in your configuration:

```yaml
litellm_settings:
  callbacks: ["prometheus"]
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check logs with `kubectl logs -f deployment/litellm -n litellm`

2. **API key errors**: Ensure secrets are properly base64 encoded

3. **Database connection issues**: Verify database service is running

4. **Load balancer not working**: Check AWS Load Balancer Controller installation

### Useful Commands

```bash
# Check pod status
kubectl get pods -n litellm

# View logs
kubectl logs -f deployment/litellm -n litellm

# Port forward for testing
kubectl port-forward svc/litellm-service 4000:80 -n litellm

# Scale deployment
kubectl scale deployment litellm --replicas=5 -n litellm
```

## AWS RDS for Production

For production deployments, use AWS RDS instead of in-cluster PostgreSQL:

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier litellm-postgres \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username litellm \
    --master-user-password your-secure-password \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-12345678 \
    --db-subnet-group-name litellm-subnet-group \
    --backup-retention-period 7 \
    --storage-encrypted
```

Update your ConfigMap to use the RDS endpoint:

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://litellm:your-password@your-rds-endpoint.rds.amazonaws.com:5432/litellm
```

## Security Best Practices

- Store API keys in Kubernetes secrets
- Use IAM roles for service accounts (IRSA)
- Enable SSL/TLS for all communications
- Regularly rotate API keys and credentials
- Use network policies to restrict traffic
- Enable audit logging

## Cost Optimization

- Use spot instances for worker nodes
- Set appropriate resource limits
- Enable auto scaling (HPA/VPA)
- Use reserved instances for predictable workloads
- Monitor usage with cost allocation tags

This guide provides everything you need to deploy LiteLLM Proxy on AWS EKS successfully. For advanced configurations, refer to the [full documentation](https://docs.litellm.ai/docs/proxy/configs).
    model_list:
      - model_name: gpt-4o
        litellm_params:
          model: openai/gpt-4o
          api_key: os.environ/OPENAI_API_KEY
      - model_name: claude-3-5-sonnet
        litellm_params:
          model: anthropic/claude-3-5-sonnet-20241022
          api_key: os.environ/ANTHROPIC_API_KEY

    general_settings:
      master_key: os.environ/LITELLM_MASTER_KEY
---
apiVersion: v1
kind: Secret
metadata:
  name: litellm-secrets
  namespace: litellm
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-openai-key>
  ANTHROPIC_API_KEY: <base64-encoded-anthropic-key>
  LITELLM_MASTER_KEY: <base64-encoded-master-key>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: litellm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
    spec:
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm:main-stable
        ports:
        - containerPort: 4000
        envFrom:
        - secretRef:
            name: litellm-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /health/liveliness
            port: 4000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: config
        configMap:
          name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-service
  namespace: litellm
spec:
  selector:
    app: litellm
  ports:
    - port: 80
      targetPort: 4000
  type: LoadBalancer
```

Apply the deployment:

```bash
kubectl apply -f litellm-deployment.yaml
```

### Step 3: Get External IP and Test

```bash
# Get the LoadBalancer URL
kubectl get svc litellm-service -n litellm

# Test the API (replace YOUR_URL with the actual LoadBalancer URL)
curl -X POST http://YOUR_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello from EKS!"}]
  }'
```

## Production Deployment

For production use, add these components:

### 1. PostgreSQL Database

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: litellm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: litellm
        - name: POSTGRES_USER
          value: litellm
        - name: POSTGRES_PASSWORD
          value: your-secure-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: litellm
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: litellm
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
```

Update your ConfigMap to include the database URL:

```yaml
data:
  config.yaml: |
    general_settings:
      master_key: os.environ/LITELLM_MASTER_KEY
      database_url: postgresql://litellm:your-secure-password@postgres-service:5432/litellm
```

### 2. Auto Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-hpa
  namespace: litellm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. Ingress with SSL

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: litellm-ingress
  namespace: litellm
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456789012:certificate/your-cert-arn
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: litellm-service
            port:
              number: 80
```

## Configuration Examples

### Multiple LLM Providers

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-02-01"

  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: os.environ/GEMINI_API_KEY
```

### Rate Limiting and Routing

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 1000
      tpm: 100000

router_settings:
  routing_strategy: usage-based-routing
  fallbacks:
    - gpt-4o: ["claude-3-5-sonnet"]

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

## Using Helm Chart

For more advanced deployments, use the official Helm chart:

```bash
# Add the Helm repository
helm repo add litellm https://berriai.github.io/litellm-helm
helm repo update

# Install with basic configuration
helm install litellm litellm/litellm \
  --namespace litellm \
  --create-namespace \
  --set masterkey=your-master-key \
  --set environmentSecrets[0]=litellm-secrets
```

## Monitoring

### Health Checks

LiteLLM provides built-in health endpoints:

```bash
# Liveliness probe
curl http://your-service/health/liveliness

# Readiness probe
curl http://your-service/health/readiness
```

### Metrics

Enable Prometheus metrics in your config:

```yaml
litellm_settings:
  callbacks: ["prometheus"]
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check logs with `kubectl logs -f deployment/litellm -n litellm`

2. **API key errors**: Ensure secrets are properly base64 encoded:
   ```bash
   echo -n "your-api-key" | base64
   ```

3. **Database connection issues**: Verify the database service is running and accessible

4. **Load balancer not working**: Check AWS Load Balancer Controller installation

### Useful Commands

```bash
# Check pod status
kubectl get pods -n litellm

# View logs
kubectl logs -f deployment/litellm -n litellm

# Describe pod for events
kubectl describe pod <pod-name> -n litellm

# Port forward for testing
kubectl port-forward svc/litellm-service 4000:80 -n litellm

# Scale deployment
kubectl scale deployment litellm --replicas=5 -n litellm
```

## Cost Optimization

- Use spot instances for worker nodes
- Set appropriate resource limits
- Enable HPA for automatic scaling
- Use reserved instances for steady workloads
- Monitor usage with cost allocation tags

## Security Best Practices

- Store API keys in Kubernetes secrets
- Use IAM roles for service accounts (IRSA)
- Enable network policies
- Use SSL/TLS for all communications
- Regularly rotate API keys and credentials
- Enable audit logging

This guide provides a production-ready deployment of LiteLLM Proxy on AWS EKS. For advanced configurations, refer to the [full documentation](https://docs.litellm.ai/docs/proxy/configs).

## Production Deployment with Database

<Tabs>
<TabItem value="postgresql" label="With PostgreSQL">

### RDS PostgreSQL Setup

1. **Create RDS Instance**:
```bash
aws rds create-db-instance \
    --db-instance-identifier litellm-postgres \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username litellm \
    --master-user-password your-secure-password \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-12345678 \
    --db-subnet-group-name litellm-subnet-group \
    --backup-retention-period 7 \
    --storage-encrypted
```

Update your ConfigMap to use the RDS endpoint:

```yaml
data:
  config.yaml: |
    general_settings:
      master_key: os.environ/LITELLM_MASTER_KEY
      database_url: postgresql://litellm:your-password@your-rds-endpoint.rds.amazonaws.com:5432/litellm
```

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: litellm-prod
---
apiVersion: v1
kind: Secret
metadata:
  name: litellm-secrets
  namespace: litellm-prod
type: Opaque
data:
  LITELLM_MASTER_KEY: your-base64-master-key
  LITELLM_SALT_KEY: your-base64-salt-key
  DATABASE_URL: your-base64-database-url
  OPENAI_API_KEY: your-base64-openai-key
  ANTHROPIC_API_KEY: your-base64-anthropic-key
  AZURE_API_KEY: your-base64-azure-key
  AZURE_API_BASE: your-base64-azure-base
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-deployment
  namespace: litellm-prod
  labels:
    app: litellm
    version: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
        version: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "4000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: litellm-service-account
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - litellm
              topologyKey: kubernetes.io/hostname
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm-database:main-stable
        ports:
        - containerPort: 4000
          name: http
        args:
          - "--port"
          - "4000"
          - "--num_workers"
          - "4"
        envFrom:
        - secretRef:
            name: litellm-secrets
        env:
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://otel-collector:4317"
        - name: AWS_REGION
          value: "us-west-2"
        livenessProbe:
          httpGet:
            path: /health/liveliness
            port: 4000
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4000
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

</TabItem>
</Tabs>

## Load Balancing and Ingress

### AWS Load Balancer Controller

1. **Install AWS Load Balancer Controller**:

```bash
# Create IAM role for AWS Load Balancer Controller
eksctl create iamserviceaccount \
  --cluster=litellm-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
  --approve

# Install via Helm
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=litellm-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

2. **Create Ingress**:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: litellm-ingress
  namespace: litellm-prod
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '30'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '3'
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-west-2:123456789012:certificate/your-cert-arn
spec:
  rules:
  - host: litellm-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: litellm-service
            port:
              number: 80
```

## Auto Scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-hpa
  namespace: litellm-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm-deployment
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
```

### Vertical Pod Autoscaler (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: litellm-vpa
  namespace: litellm-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: litellm
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
      controlledResources: ["cpu", "memory"]
```

### Cluster Autoscaler

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
      annotations:
        prometheus.io/scrape: 'true'
        prometheus.io/port: '8085'
    spec:
      serviceAccount: cluster-autoscaler
      containers:
      - image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.29.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/litellm-cluster
        env:
        - name: AWS_REGION
          value: us-west-2
```

## Security and RBAC

### Service Account and RBAC

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: litellm-service-account
  namespace: litellm-prod
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/LiteLLMServiceRole
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: litellm-prod
  name: litellm-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: litellm-rolebinding
  namespace: litellm-prod
subjects:
- kind: ServiceAccount
  name: litellm-service-account
  namespace: litellm-prod
roleRef:
  kind: Role
  name: litellm-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: litellm-network-policy
  namespace: litellm-prod
spec:
  podSelector:
    matchLabels:
      app: litellm
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 4000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP
    - protocol: UDP
      port: 53   # DNS
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Standards

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: litellm-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## Monitoring and Observability

### Prometheus and Grafana Setup

1. **Install Prometheus Stack**:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=your-secure-password
```

2. **ServiceMonitor for LiteLLM**:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: litellm-metrics
  namespace: litellm-prod
  labels:
    app: litellm
spec:
  selector:
    matchLabels:
      app: litellm
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### OpenTelemetry Setup

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: litellm-prod
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
    
    processors:
      batch:
        timeout: 1s
        send_batch_size: 1024
      memory_limiter:
        limit_mib: 512
    
    exporters:
      awsxray:
        region: us-west-2
      awscloudwatchmetrics:
        region: us-west-2
        namespace: LiteLLM/EKS
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [awsxray]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [awscloudwatchmetrics]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: litellm-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      serviceAccountName: otel-collector-service-account
      containers:
      - name: otel-collector
        image: public.ecr.aws/aws-observability/aws-otel-collector:latest
        command:
          - "/awscollector"
          - "--config=/conf/config.yaml"
        volumeMounts:
        - name: config-volume
          mountPath: /conf
        env:
        - name: AWS_REGION
          value: "us-west-2"
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 200m
            memory: 256Mi
      volumes:
      - name: config-volume
        configMap:
          name: otel-collector-config
```

## Helm Chart Deployment

### Custom Helm Chart

Create `Chart.yaml`:

```yaml
apiVersion: v2
name: litellm
description: A Helm chart for LiteLLM Proxy
type: application
version: 0.1.0
appVersion: "main-stable"
```

Create `values.yaml`:

```yaml
image:
  repository: ghcr.io/berriai/litellm-database
  tag: main-stable
  pullPolicy: IfNotPresent

replicaCount: 3

service:
  type: ClusterIP
  port: 80
  targetPort: 4000

ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
  hosts:
    - host: litellm-api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 50
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

config:
  master_key: "sk-your-master-key"
  database_url: "postgresql://user:pass@host:5432/db"

secrets:
  openai_api_key: ""
  anthropic_api_key: ""
  azure_api_key: ""
  azure_api_base: ""

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

Deploy with Helm:

```bash
helm install litellm ./litellm-chart \
  --namespace litellm-prod \
  --create-namespace \
  --values values.yaml
```

## Configuration Management

### ConfigMap with S3 Sync

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: config-sync
  namespace: litellm-prod
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: config-sync-service-account
          containers:
          - name: config-sync
            image: amazon/aws-cli:latest
            command:
            - /bin/sh
            - -c
            - |
              aws s3 cp s3://your-config-bucket/litellm-config.yaml /tmp/config.yaml
              kubectl create configmap litellm-config --from-file=/tmp/config.yaml --dry-run=client -o yaml | kubectl apply -f -
              kubectl rollout restart deployment/litellm-deployment
            env:
            - name: AWS_REGION
              value: us-west-2
          restartPolicy: OnFailure
```

### Load Testing

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: litellm-load-test
  namespace: litellm-prod
spec:
  template:
    spec:
      containers:
      - name: load-test
        image: loadimpact/k6:latest
        command:
        - k6
        - run
        - --vus=50
        - --duration=5m
        - /scripts/load-test.js
        volumeMounts:
        - name: test-scripts
          mountPath: /scripts
      volumes:
      - name: test-scripts
        configMap:
          name: load-test-scripts
      restartPolicy: Never
```

````