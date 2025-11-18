#!/usr/bin/env python3
"""
🚀 Skypiea Gateway Watchtower Status
Quick status overview of the monitoring system
"""

import requests
import subprocess
import time
from datetime import datetime

def check_server_status():
    """Check basic server status"""
    try:
        auth_header = {"Authorization": "Bearer sk-ifGZJqF3PjuEY5yFbN2B"}
        response = requests.get("http://localhost:4000/health", headers=auth_header, timeout=5)

        if response.status_code == 200:
            data = response.json()
            healthy = data.get("healthy_count", 0)
            unhealthy = data.get("unhealthy_count", 0)

            status_emoji = "✅" if unhealthy == 0 else "⚠️" if unhealthy < healthy else "❌"
            return f"{status_emoji} Server Online - {healthy} healthy, {unhealthy} unhealthy endpoints"
        else:
            return f"❌ Server responded with status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "❌ Server not reachable (connection failed)"
    except Exception as e:
        return f"❌ Error checking server: {str(e)[:50]}..."

def check_processes():
    """Check monitoring processes"""
    processes = {
        "LiteLLM Server": "litellm.*--port.*4000",
        "Watchtower Monitor": "watchtower_monitor.py",
        "Log Monitor": "log_monitor.py",
        "Dashboard": "watchtower_dashboard.py"
    }

    status = {}
    for name, pattern in processes.items():
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                count = len(result.stdout.strip().split('\n'))
                status[name] = f"✅ Running ({count} process{'es' if count > 1 else ''})"
            else:
                status[name] = "❌ Not running"
        except:
            status[name] = "❓ Check failed"

    return status

def get_system_info():
    """Get basic system information"""
    try:
        # Get server PID and basic info
        result = subprocess.run(
            ["pgrep", "-f", "litellm.*--port.*4000"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]

            # Get process info
            ps_result = subprocess.run(
                ["ps", "-p", pid, "-o", "pcpu,pmem,etime"],
                capture_output=True, text=True
            )

            if ps_result.returncode == 0:
                lines = ps_result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    values = lines[1].split()
                    if len(values) >= 3:
                        cpu = float(values[0])
                        mem = float(values[1])
                        uptime = values[2]
                        return f"PID: {pid} | CPU: {cpu:.1f}% | Memory: {mem:.1f}% | Uptime: {uptime}"

        return "Server process info unavailable"
    except:
        return "System info check failed"

def main():
    """Main status display"""
    print("=" * 60)
    print("🚀 SKYPIEA GATEWAY WATCHTOWER STATUS")
    print("=" * 60)
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Server status
    print("📊 SERVER STATUS:")
    server_status = check_server_status()
    print(f"   {server_status}")
    print()

    # System info
    print("💻 SYSTEM INFO:")
    sys_info = get_system_info()
    print(f"   {sys_info}")
    print()

    # Process status
    print("🔧 MONITORING PROCESSES:")
    processes = check_processes()
    for name, status in processes.items():
        print(f"   {name}: {status}")
    print()

    # Quick actions
    print("🎮 QUICK ACTIONS:")
    print("   • Run 'python3 watchtower_dashboard.py' for interactive dashboard")
    print("   • Run 'python3 watchtower_monitor.py' for console monitoring")
    print("   • Run 'python3 log_monitor.py' for log monitoring")
    print("   • Check 'curl -H \"Authorization: Bearer sk-ifGZJqF3PjuEY5yFbN2B\" http://localhost:4000/health' for health")
    print()

    # Known issues reminder
    print("⚠️  KNOWN ISSUES:")
    print("   • OpenRouter credits may need replenishment")
    print("   • Some models have authentication issues")
    print("   • Check OpenRouter dashboard: https://openrouter.ai/settings/credits")

    print("=" * 60)

if __name__ == "__main__":
    main()
