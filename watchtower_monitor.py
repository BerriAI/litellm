#!/usr/bin/env python3
"""
🚀 Skypiea Gateway Watchtower Monitor
Monitors the LiteLLM proxy server health, performance, and logs
"""

import time
import requests
import psutil
import json
import subprocess
import threading
from datetime import datetime
import sys
import os

class WatchtowerMonitor:
    def __init__(self, host="localhost", port=4000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = None
        self.server_pid = None
        self.auth_header = {"Authorization": "Bearer sk-ifGZJqF3PjuEY5yFbN2B"}
        self.alerts = []
        self.known_issues = set()

    def get_server_pid(self):
        """Find the litellm server process PID"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "litellm.*--port.*4000"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return None

    def check_server_health(self):
        """Check basic server connectivity and model health"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.auth_header, timeout=10)
            health_data = response.json()

            self.last_health_check = {
                "status_code": response.status_code,
                "healthy_count": health_data.get("healthy_count", 0),
                "unhealthy_count": health_data.get("unhealthy_count", 0),
                "timestamp": datetime.now().isoformat()
            }

            # Check for new issues
            if health_data.get("unhealthy_count", 0) > 0:
                for endpoint in health_data.get("unhealthy_endpoints", []):
                    error_msg = endpoint.get("error", "")
                    issue_key = f"{endpoint.get('model', 'unknown')}:{endpoint.get('custom_llm_provider', 'unknown')}"

                    if "Insufficient credits" in error_msg and issue_key not in self.known_issues:
                        self.alerts.append({
                            "type": "CREDITS",
                            "model": endpoint.get("model"),
                            "message": "OpenRouter account has insufficient credits",
                            "timestamp": datetime.now().isoformat()
                        })
                        self.known_issues.add(issue_key)

                    elif "No auth credentials found" in error_msg and issue_key not in self.known_issues:
                        self.alerts.append({
                            "type": "AUTH",
                            "model": endpoint.get("model"),
                            "message": "Missing authentication credentials",
                            "timestamp": datetime.now().isoformat()
                        })
                        self.known_issues.add(issue_key)

            return response.status_code == 200
        except Exception as e:
            self.last_health_check = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            if "server_down" not in self.known_issues:
                self.alerts.append({
                    "type": "SERVER",
                    "message": f"Server health check failed: {e}",
                    "timestamp": datetime.now().isoformat()
                })
                self.known_issues.add("server_down")
            return False

    def check_models_endpoint(self):
        """Check /models endpoint"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.auth_header, timeout=10)
            data = response.json()
            models = data.get("data", [])
            return {
                "status_code": response.status_code,
                "models_count": len(models),
                "models": [model.get("id") for model in models],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_system_resources(self):
        """Get system resource usage"""
        if self.server_pid:
            try:
                process = psutil.Process(self.server_pid)
                cpu_percent = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()

                return {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "memory_percent": memory_percent,
                    "timestamp": datetime.now().isoformat()
                }
            except:
                pass

        return {"error": "Could not get process info", "timestamp": datetime.now().isoformat()}

    def test_chat_completion(self):
        """Test a simple chat completion"""
        try:
            payload = {
                "model": "claude-3.5-sonnet",
                "messages": [{"role": "user", "content": "Hello, this is a watchtower test. Please respond with 'OK' only."}],
                "max_tokens": 5
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self.auth_header,
                timeout=30
            )
            self.request_count += 1
            success = response.status_code == 200
            if success:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "status_code": response.status_code,
                    "success": True,
                    "response": content.strip(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status_code": response.status_code,
                    "success": False,
                    "error": response.text[:200],
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.error_count += 1
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def print_status(self):
        """Print current monitoring status"""
        uptime = datetime.now() - self.start_time

        print(f"\n{'='*60}")
        print(f"🚀 SKYPIEA GATEWAY WATCHTOWER - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"📊 Server Status: {self.base_url}")
        print(f"⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f"🔢 Requests: {self.request_count} | Errors: {self.error_count}")

        # Health check status
        if self.last_health_check:
            if "status_code" in self.last_health_check and "healthy_count" in self.last_health_check:
                healthy = self.last_health_check['healthy_count']
                unhealthy = self.last_health_check['unhealthy_count']
                status_emoji = "✅" if unhealthy == 0 else "⚠️" if unhealthy < healthy else "❌"
                print(f"💚 Health: {status_emoji} (HTTP {self.last_health_check['status_code']}) - {healthy} healthy, {unhealthy} unhealthy")
            elif "error" in self.last_health_check:
                print(f"💔 Health: ❌ {self.last_health_check.get('error', 'Unknown error')}")

        # Process info
        if self.server_pid:
            print(f"🔧 Process PID: {self.server_pid}")
        else:
            print("🔧 Process PID: Not found")

        # Show recent alerts
        if self.alerts:
            recent_alerts = [a for a in self.alerts[-3:]]  # Show last 3 alerts
            print(f"\n🚨 Recent Alerts ({len(self.alerts)} total):")
            for alert in recent_alerts:
                alert_type = alert.get('type', 'UNKNOWN')
                emoji = {"CREDITS": "💰", "AUTH": "🔐", "SERVER": "🔥"}.get(alert_type, "⚠️")
                print(f"  {emoji} [{alert_type}] {alert.get('message', 'Unknown alert')}")
                if 'model' in alert:
                    print(f"     Model: {alert['model']}")

    def monitor_loop(self, interval=30):
        """Main monitoring loop"""
        print("🚀 Starting Skypiea Gateway Watchtower Monitor...")

        while True:
            try:
                # Update PID if needed
                if not self.server_pid:
                    self.server_pid = self.get_server_pid()

                # Run checks
                health_ok = self.check_server_health()
                models_info = self.check_models_endpoint()
                resources = self.get_system_resources()

                # Print status
                self.print_status()

                # Detailed info
                print("\n📋 Details:")
                if health_ok:
                    print("✅ Server responding to health checks")
                else:
                    print("❌ Server health check failed")

                if "models_count" in models_info and "models" in models_info:
                    print(f"🤖 Models available: {models_info['models_count']}")
                    print(f"   Models: {', '.join(models_info['models'][:3])}{'...' if len(models_info['models']) > 3 else ''}")
                elif "error" in models_info:
                    print(f"🤖 Models check error: {models_info['error']}")

                if "cpu_percent" in resources:
                    print(".1f"                elif "error" in resources:
                    print(f"💻 Resources: {resources['error']}")

                # Test API occasionally (every 2 minutes)
                if int(time.time()) % 120 == 0:
                    print("\n🧪 Testing API endpoint...")
                    test_result = self.test_chat_completion()
                    if test_result.get("success"):
                        response_text = test_result.get("response", "")
                        print(f"✅ Chat completion test passed: '{response_text}'")
                    else:
                        error_msg = test_result.get("error", "Unknown error")
                        print(f"❌ Chat completion test failed: {error_msg[:100]}...")
                        if "Insufficient credits" in error_msg:
                            print("💰 Note: OpenRouter credits may need to be replenished")

                # Show environment info on first run
                if int(time.time()) % 600 == 0:  # Every 10 minutes
                    print("\n🔧 System Information:")
                    print(f"   Python: {sys.version.split()[0]}")
                    print(f"   Platform: {sys.platform}")
                    try:
                        import litellm
                        print(f"   LiteLLM version: {litellm.__version__}")
                    except:
                        print("   LiteLLM version: Unknown")

            except Exception as e:
                print(f"❌ Monitoring error: {e}")

            time.sleep(interval)

if __name__ == "__main__":
    monitor = WatchtowerMonitor()
    monitor.monitor_loop()
