#!/usr/bin/env python3
"""
🚀 Skypiea Gateway Watchtower Dashboard
Real-time monitoring dashboard for the LiteLLM server
"""

import os
import time
import requests
import subprocess
import json
from datetime import datetime
import curses
import threading
from typing import Dict, Any

class WatchtowerDashboard:
    def __init__(self):
        self.host = "localhost"
        self.port = 4000
        self.base_url = f"http://{self.host}:{self.port}"
        self.auth_header = {"Authorization": "Bearer sk-ifGZJqF3PjuEY5yFbN2B"}
        self.data = {
            "server_status": "Checking...",
            "uptime": "Unknown",
            "models_count": 0,
            "healthy_endpoints": 0,
            "unhealthy_endpoints": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "requests_total": 0,
            "errors_total": 0,
            "last_health_check": None,
            "alerts": [],
            "recent_logs": []
        }
        self.running = True

    def get_server_pid(self) -> int:
        """Get the server process ID"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "litellm.*--port.*4000"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return 0

    def update_server_status(self):
        """Update server status information"""
        try:
            # Check health
            response = requests.get(f"{self.base_url}/health", headers=self.auth_header, timeout=5)
            health_data = response.json()

            self.data["server_status"] = "Running"
            self.data["healthy_endpoints"] = health_data.get("healthy_count", 0)
            self.data["unhealthy_endpoints"] = health_data.get("unhealthy_count", 0)
            self.data["last_health_check"] = datetime.now().strftime("%H:%M:%S")

            # Check for alerts
            if health_data.get("unhealthy_count", 0) > 0:
                for endpoint in health_data.get("unhealthy_endpoints", []):
                    error_msg = endpoint.get("error", "")
                    model = endpoint.get("model", "unknown")

                    if "Insufficient credits" in error_msg and model not in [a.get("model") for a in self.data["alerts"]]:
                        self.data["alerts"].append({
                            "type": "CREDITS",
                            "model": model,
                            "message": "OpenRouter credits insufficient",
                            "time": datetime.now().strftime("%H:%M:%S")
                        })

        except Exception as e:
            self.data["server_status"] = f"Error: {str(e)[:30]}..."

    def update_models_info(self):
        """Update models information"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.auth_header, timeout=5)
            data = response.json()
            self.data["models_count"] = len(data.get("data", []))
        except:
            self.data["models_count"] = 0

    def update_system_resources(self):
        """Update system resource usage"""
        pid = self.get_server_pid()
        if pid:
            try:
                # Simple CPU/memory check using ps
                result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "pcpu,pmem"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        values = lines[1].split()
                        if len(values) >= 2:
                            self.data["cpu_usage"] = float(values[0])
                            self.data["memory_usage"] = float(values[1])
            except:
                pass

    def update_uptime(self):
        """Update server uptime"""
        pid = self.get_server_pid()
        if pid:
            try:
                result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "etime"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        self.data["uptime"] = lines[1].strip()
            except:
                pass

    def draw_dashboard(self, stdscr):
        """Draw the monitoring dashboard"""
        stdscr.clear()

        # Colors
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Healthy
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Info

        height, width = stdscr.getmaxyx()

        # Title
        title = "🚀 SKYPIEA GATEWAY WATCHTOWER DASHBOARD"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        # Time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stdscr.addstr(1, width - len(current_time) - 1, current_time, curses.color_pair(4))

        line = 3

        # Server Status Section
        stdscr.addstr(line, 2, "📊 SERVER STATUS", curses.A_BOLD)
        line += 1

        status_color = curses.color_pair(1) if self.data["server_status"] == "Running" else curses.color_pair(2)
        stdscr.addstr(line, 4, f"Status: {self.data['server_status']}", status_color)
        line += 1

        stdscr.addstr(line, 4, f"Uptime: {self.data['uptime']}", curses.color_pair(4))
        line += 1

        stdscr.addstr(line, 4, f"Last Health Check: {self.data['last_health_check'] or 'Never'}", curses.color_pair(4))
        line += 2

        # Health Section
        stdscr.addstr(line, 2, "💚 ENDPOINT HEALTH", curses.A_BOLD)
        line += 1

        healthy_color = curses.color_pair(1) if self.data["unhealthy_endpoints"] == 0 else curses.color_pair(3)
        stdscr.addstr(line, 4, f"Healthy: {self.data['healthy_endpoints']}", healthy_color)
        line += 1

        unhealthy_color = curses.color_pair(2) if self.data["unhealthy_endpoints"] > 0 else curses.color_pair(1)
        stdscr.addstr(line, 4, f"Unhealthy: {self.data['unhealthy_endpoints']}", unhealthy_color)
        line += 2

        # Models Section
        stdscr.addstr(line, 2, "🤖 MODELS", curses.A_BOLD)
        line += 1

        stdscr.addstr(line, 4, f"Available: {self.data['models_count']}", curses.color_pair(4))
        line += 2

        # Resources Section
        stdscr.addstr(line, 2, "💻 RESOURCES", curses.A_BOLD)
        line += 1

        stdscr.addstr(line, 4, f"CPU: {self.data['cpu_usage']:.1f}%", curses.color_pair(4))
        line += 1

        stdscr.addstr(line, 4, f"Memory: {self.data['memory_usage']:.1f}%", curses.color_pair(4))
        line += 2

        # Alerts Section
        if self.data["alerts"]:
            stdscr.addstr(line, 2, "🚨 ACTIVE ALERTS", curses.A_BOLD | curses.color_pair(2))
            line += 1

            for i, alert in enumerate(self.data["alerts"][-3:]):  # Show last 3
                if line + i < height - 2:
                    alert_type = alert.get('type', 'UNKNOWN')
                    emoji = {"CREDITS": "💰", "AUTH": "🔐", "SERVER": "🔥"}.get(alert_type, "⚠️")
                    alert_text = f"{emoji} {alert.get('message', 'Unknown')} ({alert.get('time', '')})"
                    stdscr.addstr(line + i, 4, alert_text[:width-6], curses.color_pair(2))

        # Footer
        footer = "Press 'q' to quit | Press 'r' to refresh"
        stdscr.addstr(height - 1, (width - len(footer)) // 2, footer, curses.color_pair(3))

        stdscr.refresh()

    def data_collection_loop(self):
        """Background thread for data collection"""
        while self.running:
            try:
                self.update_server_status()
                self.update_models_info()
                self.update_system_resources()
                self.update_uptime()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                pass

    def run(self, stdscr):
        """Main dashboard loop"""
        # Start data collection thread
        data_thread = threading.Thread(target=self.data_collection_loop, daemon=True)
        data_thread.start()

        # Set up curses
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(1000)  # Refresh every second

        while self.running:
            try:
                self.draw_dashboard(stdscr)

                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    # Force refresh
                    pass

            except KeyboardInterrupt:
                self.running = False
                break

def main():
    """Main function"""
    print("🚀 Starting Skypiea Gateway Watchtower Dashboard...")
    print("Press 'q' to quit, 'r' to refresh")
    time.sleep(1)

    dashboard = WatchtowerDashboard()

    try:
        curses.wrapper(dashboard.run)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.running = False
        print("\n👋 Watchtower Dashboard closed")

if __name__ == "__main__":
    main()
