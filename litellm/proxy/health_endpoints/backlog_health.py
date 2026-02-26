import platform
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict


class BacklogHealth:
    cache: Dict[str, Any] = {
        "last_updated": datetime.min,
        "port": None,
        "data": None,
    }

    @staticmethod
    def parse_linux_ss_listen_queue(output: str, port: int) -> Dict[str, Any]:
        for line in output.splitlines():
            if "LISTEN" not in line or f":{port}" not in line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            recv_q_str = parts[1]
            send_q_str = parts[2]
            if recv_q_str.isdigit() and send_q_str.isdigit():
                return {
                    "listen_queue_current": int(recv_q_str),
                    "listen_queue_max": int(send_q_str),
                }
        raise ValueError(
            f"Could not parse listen queue metrics from ss output for port {port}"
        )

    @staticmethod
    def parse_macos_netstat_listen_queue(output: str, port: int) -> Dict[str, Any]:
        port_regex = re.compile(rf"(?:\.|:){port}\b")
        queue_regex = re.compile(r"(?:^|\s)(\d+)/(\d+)/(\d+)(?:\s|$)")

        for line in output.splitlines():
            if not port_regex.search(line):
                continue
            queue_match = queue_regex.search(line)
            if queue_match is None:
                continue
            current = int(queue_match.group(1))
            max_size = int(queue_match.group(3))
            return {
                "listen_queue_current": current,
                "listen_queue_max": max_size,
            }
        raise ValueError(
            f"Could not parse listen queue metrics from netstat output for port {port}"
        )

    @staticmethod
    def read_listen_queue_stats_for_port(port: int) -> Dict[str, Any]:
        system = platform.system().lower()
        sampled_at = datetime.utcnow().isoformat() + "Z"

        try:
            if system == "linux":
                command = ["ss", "-ltn", "sport", "=", f":{port}"]
                output = subprocess.check_output(command, text=True)
                parsed = BacklogHealth.parse_linux_ss_listen_queue(
                    output=output, port=port
                )
                source = "ss"
            elif system in ("darwin", "freebsd"):
                command = ["netstat", "-Lan"]
                output = subprocess.check_output(command, text=True)
                parsed = BacklogHealth.parse_macos_netstat_listen_queue(
                    output=output, port=port
                )
                source = "netstat -Lan"
            else:
                return {
                    "status": "unavailable",
                    "error": f"Unsupported platform for listen queue monitoring: {system}",
                    "port": port,
                    "sampled_at": sampled_at,
                }

            current = parsed["listen_queue_current"]
            max_size = parsed["listen_queue_max"]
            utilization = (
                round(float(current) / float(max_size), 4)
                if max_size and max_size > 0
                else 0.0
            )

            pressure_status = "ok"
            if utilization >= 0.9:
                pressure_status = "saturated"
            elif utilization >= 0.7:
                pressure_status = "degraded"

            return {
                "status": "healthy",
                "port": port,
                "source": source,
                "listen_queue_current": current,
                "listen_queue_max": max_size,
                "listen_queue_utilization": utilization,
                "listen_queue_status": pressure_status,
                "sampled_at": sampled_at,
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e),
                "port": port,
                "sampled_at": sampled_at,
            }

    @staticmethod
    def get_cached_listen_queue_stats(port: int, ttl_seconds: int = 5) -> Dict[str, Any]:
        now = datetime.now()
        cache_port = BacklogHealth.cache.get("port")
        cache_data = BacklogHealth.cache.get("data")
        last_updated = BacklogHealth.cache.get("last_updated")

        if (
            cache_port == port
            and cache_data is not None
            and isinstance(last_updated, datetime)
            and (now - last_updated) < timedelta(seconds=ttl_seconds)
        ):
            return cache_data

        sampled = BacklogHealth.read_listen_queue_stats_for_port(port=port)
        BacklogHealth.cache = {"last_updated": now, "port": port, "data": sampled}
        return sampled
