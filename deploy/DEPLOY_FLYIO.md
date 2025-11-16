# LiteLLM Deployment on Fly.io

This guide provides instructions for deploying LiteLLM to Fly.io using the secure, non-root container environment.

## Prerequisites

- [Fly.io account](https://fly.io)
- [flyctl CLI installed](/install-flyctl)
- LiteLLM repository cloned locally

## Benefits of Non-Root Container

The `Dockerfile.non_root` provides:
- **Enhanced Security**: Runs as `nobody` user (UID 65534) instead of root
- **Minimal Base Image**: Uses Chainguard Python images for smaller attack surface
- **Compliance**: Meets security best practices and compliance requirements
- **Optimized**: Multi-stage builds reduce image size

## Deployment Steps

### 1. Initialize Fly.io App

First, authenticate with Fly.io:

```bash
flyctl auth login
```

The `fly.toml` configuration file is already set up in the repository. To launch your app:

```bash
flyctl launch
```

During launch, Fly.io will:
- Detect the `fly.toml` configuration
- Use `docker/Dockerfile.non_root` for building
- Create a new Fly.io application

### 2. Set Required Secrets

Set the `LITELLM_MASTER_KEY` (required for token signing):

```bash
flyctl secrets set LITELLM_MASTER_KEY=your-super-secret-key-here
```

Generate a secure key with:
```bash
openssl rand -hex 32
```

### 3. Set Optional Environment Variables

If using Fly Postgres, set the DATABASE_URL:

```bash
flyctl secrets set DATABASE_URL="postgres://user:password@your-db-host:5432/litellm"
```

### 4. Mount Configuration Files

#### Build Custom Docker Image with Embedded Config

Use `deploy/Dockerfile.ghcr_base` that copies your config.

### 5. Deploy

Deploy your application:

```bash
flyctl deploy
```

Or redeploy with new code:

```bash
git push  # Push your changes
flyctl deploy
```

### 6. Monitor Deployment

Check deployment status:

```bash
flyctl status
```

View logs:

```bash
flyctl logs
```

View logs in real-time:

```bash
flyctl logs -f
```

## Configuration Details

### `fly.toml` Settings

- **Region**: Set to `nrt` (Tokyo) by default, changeable via `primary_region`
- **VM Size**: `shared-cpu-1x` with 256 MB memory (adjustable)
- **Swap Size**: 512 MB for additional memory buffer
- **Volumes**: 
  - `litellm_config` (1 GB): For `config.yaml`
  - `litellm_env` (1 GB): For `.env` file
- **Processes**: App runs on port 4000, exposed via HTTP/HTTPS
- **Health Checks**: Configured to check `/health/liveliness` every 15 seconds
- **Port**: Container runs on port 4000, exposed via HTTP/HTTPS
- **Health Checks**: Configured to check `/health/liveliness` every 15 seconds
- **Volumes**: Persistent storage for logs and temporary files
- **Database Migrations**: Automatic `prisma migrate deploy` on release

### Scaling

Scale the number of machines:

```bash
flyctl scale count 3
```

Scale machine size:

```bash
flyctl machine update <id> --vm-size shared-cpu-4x
```

### Environment Variables

Common environment variables you may want to set:

```bash
# Allow adding models via UI
flyctl secrets set STORE_MODEL_IN_DB=True

# Customize proxy behavior
flyctl secrets set LOG_LEVEL=INFO
```

## Database Setup

### Option 1: Fly Postgres

Create a Postgres database:

```bash
flyctl postgres create
```

Attach to your LiteLLM app:

```bash
flyctl postgres attach litellm
```

### Option 2: External Database

Set the connection string:

```bash
flyctl secrets set DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

## Health Checks and Monitoring

The deployment includes HTTP health checks that:
- Check every 15 seconds
- Require 30 seconds grace period after deployment
- Hit the `/health/liveliness` endpoint
- Timeout after 5 seconds

Monitor machine health:

```bash
flyctl status
```

## Volume Management

Two persistent volumes are created:

1. **litellm_tmp** (5 GB): Temporary files
2. **litellm_logs** (10 GB): Application logs

Check volume usage:

```bash
flyctl volumes list
```

Resize a volume:

```bash
flyctl volumes extend <volume-id> --size 20
```

## Updating Configuration

To update environment variables:

```bash
flyctl secrets set VAR_NAME=value
```

To update Dockerfile references:

```bash
flyctl deploy
```

## Troubleshooting

### Container won't start

Check logs:
```bash
flyctl logs --err
```

Verify health checks:
```bash
flyctl status
```

### Permission denied errors

Ensure the Dockerfile runs as `nobody` user. The `Dockerfile.non_root` is already configured for this.

### Database connection issues

Verify DATABASE_URL is set:
```bash
flyctl secrets list
```

Test connection from machine:
```bash
flyctl ssh console
```

### Disk space issues

Check volume status:
```bash
flyctl volumes list
```

Extend volumes as needed:
```bash
flyctl volumes extend <volume-id> --size 50
```

## Useful Commands

```bash
# SSH into running machine
flyctl ssh console

# Restart all machines
flyctl machines restart

# View all machines
flyctl machines list

# Stop deployment
flyctl scale count 0

# Resume deployment
flyctl scale count 1

# View app metrics
flyctl metrics

# Run command in machine
flyctl ssh console -s -- <command>
```

## Security Best Practices

1. **Rotate Secrets**: Regularly update `MASTER_KEY`
   ```bash
   flyctl secrets set MASTER_KEY=new-secret-key
   ```

2. **Network Security**: LiteLLM is exposed publicly. Consider:
   - Using IP allowlists
   - Requiring authentication tokens
   - Rate limiting via proxy configuration

3. **Updates**: Keep LiteLLM updated:
   ```bash
   git pull
   flyctl deploy
   ```

4. **Non-Root Execution**: The non-root container runs as `nobody` user for improved security

## Additional Resources

- [Fly.io Documentation](https://fly.io/docs/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
