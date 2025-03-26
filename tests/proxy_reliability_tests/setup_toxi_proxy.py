import os
import platform
import subprocess
import time
import requests
import stat


def download_toxiproxy():
    # Determine system architecture and OS
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map architecture names
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }

    arch = arch_map.get(machine, machine)

    # Construct download URL (using latest version 2.5.0)
    base_url = "https://github.com/Shopify/toxiproxy/releases/download/v2.5.0/"
    if system == "linux":
        filename = f"toxiproxy-server-linux-{arch}"
        cli_filename = f"toxiproxy-cli-linux-{arch}"
    elif system == "darwin":
        filename = f"toxiproxy-server-darwin-{arch}"
        cli_filename = f"toxiproxy-cli-darwin-{arch}"
    else:
        raise Exception("Unsupported operating system")

    # Download server
    response = requests.get(f"{base_url}{filename}")
    with open("toxiproxy-server", "wb") as f:
        f.write(response.content)

    # Download CLI
    response = requests.get(f"{base_url}{cli_filename}")
    with open("toxiproxy-cli", "wb") as f:
        f.write(response.content)

    # Make files executable
    os.chmod("toxiproxy-server", stat.S_IRWXU)
    os.chmod("toxiproxy-cli", stat.S_IRWXU)


def setup_toxiproxy():
    # Start toxiproxy-server
    server_process = subprocess.Popen(["./toxiproxy-server"])

    # Wait for server to start
    time.sleep(2)

    # Create proxy
    subprocess.run(
        [
            "./toxiproxy-cli",
            "create",
            "-l",
            "0.0.0.0:6666",
            "-u",
            "ep-dry-paper-a69g2y1q-pooler.us-west-2.aws.neon.tech:5432",
            "postgres_proxy",
        ]
    )

    return server_process


def main():
    try:
        # Download ToxiProxy binaries
        download_toxiproxy()

        # Setup ToxiProxy
        server_process = setup_toxiproxy()

        print("ToxiProxy setup completed successfully!")
        print("Proxy 'postgres_proxy' created and listening on 127.0.0.1:6666")

        # Keep the script running to maintain the proxy
        try:
            server_process.wait()
        except KeyboardInterrupt:
            server_process.terminate()
            print("\nToxiProxy server stopped")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
