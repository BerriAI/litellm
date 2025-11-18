#!/usr/bin/env python3
"""
🚀 Skypiea Gateway Log Monitor
Monitors server logs and extracts important events
"""

import subprocess
import re
import time
from datetime import datetime
import threading

class LogMonitor:
    def __init__(self):
        self.log_patterns = {
            'error': re.compile(r'(?i)(error|exception|failed|unauthorized|forbidden)'),
            'warning': re.compile(r'(?i)(warning|warn)'),
            'info': re.compile(r'(?i)(info|starting|started|listening)'),
            'request': re.compile(r'(?i)(POST|GET|PUT|DELETE)\s+(/\w+)'),
            'health': re.compile(r'(?i)(health.*check|model.*health)'),
        }
        self.events = []
        self.monitoring = False

    def monitor_logs(self):
        """Monitor server logs in real-time"""
        try:
            # Find the litellm process
            result = subprocess.run(
                ["pgrep", "-f", "litellm.*--port.*4000"],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print("❌ No litellm server process found")
                return

            pid = result.stdout.strip().split('\n')[0]
            print(f"📋 Monitoring logs for PID: {pid}")

            # Use lsof to check if process has open log files
            try:
                lsof_result = subprocess.run(
                    ["lsof", "-p", pid],
                    capture_output=True, text=True, timeout=5
                )

                log_files = []
                for line in lsof_result.stdout.split('\n'):
                    if '.log' in line or 'log' in line.lower():
                        parts = line.split()
                        if len(parts) >= 9:
                            log_files.append(parts[8])

                if log_files:
                    print(f"📄 Found log files: {log_files}")
                    # Could monitor these files if they exist
                else:
                    print("📄 No log files found - server may be logging to stdout/stderr")

            except subprocess.TimeoutExpired:
                print("⏰ lsof command timed out")
            except Exception as e:
                print(f"❌ Error checking log files: {e}")

        except Exception as e:
            print(f"❌ Log monitoring error: {e}")

    def check_recent_logs(self):
        """Check for recent log entries using journalctl or other methods"""
        try:
            # Try to get recent system logs that might contain our process
            result = subprocess.run(
                ["log", "show", "--since", "1 hour", "--grep", "litellm"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                print(f"📋 Found {len(lines)} recent log entries")

                for line in lines[-5:]:  # Show last 5 entries
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] {line}")

        except subprocess.TimeoutExpired:
            print("⏰ Log check timed out")
        except Exception as e:
            # log command might not be available on macOS
            if "log" in str(e):
                print("📋 Log monitoring not available on this system")
            else:
                print(f"❌ Error checking logs: {e}")

    def monitor_loop(self, interval=60):
        """Main monitoring loop"""
        print("🚀 Starting Skypiea Gateway Log Monitor...")

        while True:
            try:
                self.monitor_logs()
                self.check_recent_logs()

                # Check for server process status
                result = subprocess.run(
                    ["pgrep", "-f", "litellm.*--port.*4000"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    pid = result.stdout.strip().split('\n')[0]
                    print(f"✅ Server running (PID: {pid})")
                else:
                    print("❌ Server process not found!")

                print(f"⏰ Next check in {interval} seconds...\n")
                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n🛑 Log monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = LogMonitor()
    monitor.monitor_loop()
