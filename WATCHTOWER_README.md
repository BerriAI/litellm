# 🚀 Skypiea Gateway Watchtower

A comprehensive monitoring and observability system for the LiteLLM proxy server.

## Overview

The Skypiea Gateway Watchtower provides real-time monitoring, alerting, and observability for your LiteLLM deployment. It continuously monitors server health, API endpoints, system resources, and provides alerts for critical issues.

## Features

### ✅ Core Monitoring
- **Server Health**: Continuous health checks with detailed endpoint status
- **API Endpoints**: Monitoring of `/health`, `/v1/models`, and `/v1/chat/completions`
- **System Resources**: CPU, memory, and process monitoring
- **Authentication**: Built-in support for LiteLLM master key authentication

### 🚨 Intelligent Alerting
- **Credit Alerts**: Detects insufficient OpenRouter credits
- **Auth Alerts**: Identifies authentication credential issues
- **Server Alerts**: Monitors server availability and connectivity
- **Error Tracking**: Comprehensive error detection and reporting

### 📊 Monitoring Components

#### 1. Watchtower Monitor (`watchtower_monitor.py`)
Console-based monitoring with periodic status updates:
```bash
python3 watchtower_monitor.py
```

**Features:**
- Real-time health status display
- Resource usage monitoring
- API endpoint testing (every 2 minutes)
- Alert system with deduplication
- System information display

#### 2. Log Monitor (`log_monitor.py`)
Server log monitoring and event extraction:
```bash
python3 log_monitor.py
```

**Features:**
- Process identification and tracking
- Log file discovery
- Pattern-based event extraction
- System log integration

#### 3. Interactive Dashboard (`watchtower_dashboard.py`)
Curses-based real-time dashboard:
```bash
python3 watchtower_dashboard.py
```

**Features:**
- Real-time visual status display
- Color-coded health indicators
- Keyboard controls (q to quit, r to refresh)
- Live resource monitoring

#### 4. Status Overview (`watchtower_status.py`)
Quick status snapshot:
```bash
python3 watchtower_status.py
```

**Features:**
- One-shot status check
- Process enumeration
- Quick action suggestions
- Known issues summary

## Installation & Setup

### Prerequisites
- Python 3.8+
- Running LiteLLM server on port 4000
- Required packages: `requests`, `psutil`

### Quick Start
1. Ensure LiteLLM server is running:
```bash
source .venv/bin/activate
litellm --port 4000 --detailed_debug --host 0.0.0.0
```

2. Start monitoring:
```bash
# Start console monitor
python3 watchtower_monitor.py &

# Start log monitor
python3 log_monitor.py &

# View interactive dashboard
python3 watchtower_dashboard.py
```

## Configuration

### Authentication
The watchtower automatically uses the master key from your `.env` file:
```
LITELLM_MASTER_KEY="sk-ifGZJqF3PjuEY5yFbN2B"
```

### Server Configuration
- **Host**: `localhost` (configurable)
- **Port**: `4000` (configurable)
- **Check Interval**: 30 seconds (configurable)

## Monitoring Metrics

### Server Health
- HTTP status codes
- Healthy vs unhealthy endpoints
- Response times
- Authentication status

### System Resources
- CPU usage percentage
- Memory usage percentage
- Process uptime
- Process ID tracking

### API Endpoints
- `/health`: Model health status
- `/v1/models`: Available models count
- `/v1/chat/completions`: API functionality tests

### Alerts & Issues
- **CREDITS**: OpenRouter credit depletion
- **AUTH**: Authentication credential problems
- **SERVER**: Server connectivity issues

## Current System Status

```
📊 SERVER STATUS: ⚠️ Server Online - 0 healthy, 3 unhealthy endpoints
💻 SYSTEM INFO: PID: 30307 | CPU: 0.0% | Memory: 0.0% | Uptime: 04:53
🔧 PROCESSES: LiteLLM Server ✅, Monitoring Systems ✅
```

## Known Issues & Alerts

### Active Issues
1. **OpenRouter Credits**: Insufficient credits for Claude-3.5-sonnet
2. **Authentication**: Missing credentials for some models
3. **Model Health**: 3 unhealthy endpoints detected

### Resolution Steps
1. **Credits**: Visit https://openrouter.ai/settings/credits to add credits
2. **Auth**: Check API keys in configuration
3. **Models**: Verify model configurations in `config.yaml`

## Usage Examples

### Continuous Monitoring
```bash
# Start all monitoring systems
python3 watchtower_monitor.py &
python3 log_monitor.py &
python3 watchtower_dashboard.py  # Interactive
```

### Health Check
```bash
curl -H "Authorization: Bearer sk-ifGZJqF3PjuEY5yFbN2B" \
     http://localhost:4000/health
```

### API Test
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
     -H "Authorization: Bearer sk-ifGZJqF3PjuEY5yFbN2B" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "claude-3.5-sonnet",
       "messages": [{"role": "user", "content": "Hello"}]
     }'
```

## Architecture

```
🚀 Skypiea Gateway
├── LiteLLM Proxy Server (Port 4000)
│   ├── Authentication (Master Key)
│   ├── Model Routing (OpenRouter, etc.)
│   └── Health Checks
│
└── Watchtower Monitoring System
    ├── Watchtower Monitor (Console)
    ├── Log Monitor (System Logs)
    ├── Interactive Dashboard (Curses UI)
    └── Status Script (Quick Overview)
```

## Troubleshooting

### Server Not Found
- Check if LiteLLM is running: `ps aux | grep litellm`
- Verify port 4000 is accessible: `netstat -tlnp | grep 4000`

### Authentication Errors
- Verify master key in `.env` file
- Check key format: `Bearer sk-...`

### Monitoring Not Working
- Ensure Python dependencies: `pip install requests psutil`
- Check file permissions for log access

## Development

### Adding New Monitors
1. Create new monitor class inheriting from base monitor
2. Implement `check_*` methods
3. Add to main monitoring loop
4. Update status display

### Extending Alerts
Add new alert types in the monitoring logic:
```python
if "new_condition" in error_msg:
    self.alerts.append({
        "type": "NEW_TYPE",
        "model": model,
        "message": "New alert message",
        "timestamp": datetime.now().isoformat()
    })
```

## Contributing

1. Follow existing code patterns
2. Add comprehensive error handling
3. Update documentation
4. Test with multiple server configurations

## License

Same as LiteLLM project.

---

**Watchtower Status**: 🟡 Monitoring Active | ⚠️ Issues Detected
**Last Updated**: 2025-11-18
**Version**: 1.0.0-watchtower
