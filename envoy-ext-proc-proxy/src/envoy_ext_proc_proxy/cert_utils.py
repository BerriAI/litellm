"""SSL certificate utilities for loading and generating certificates."""

import os
import ssl
import subprocess
import tempfile

import grpc


def generate_self_signed_cert(
    cert_path: str | None = None,
    key_path: str | None = None,
    hostname: str = "localhost",
    days: int = 365,
) -> tuple[str, str]:
    """Generate a self-signed certificate and private key using openssl CLI.

    If cert_path and key_path are not provided, files will be created in a
    temporary directory and will persist for the process lifetime.

    Args:
        cert_path: Optional path where the certificate PEM will be saved.
        key_path: Optional path where the private key PEM will be saved.
        hostname: Common Name / SAN hostname (default: localhost).
        days: Certificate validity period in days.

    Returns:
        Tuple of (cert_path, key_path).
    """
    if not cert_path or not key_path:
        temp_dir = tempfile.mkdtemp(prefix="proxy_tls_")
        cert_path = cert_path or os.path.join(temp_dir, "cert.pem")
        key_path = key_path or os.path.join(temp_dir, "key.pem")

    # Generate openssl config with Subject Alternative Names (SAN)
    san_entries = [
        f"DNS:{hostname}",
        "DNS:localhost",
        "IP:127.0.0.1",
        "IP:::1",
    ]
    # Filter unique SAN entries
    san_string = ",".join(sorted(set(san_entries)))

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        key_path,
        "-out",
        cert_path,
        "-days",
        str(days),
        "-subj",
        f"/CN={hostname}",
        "-addext",
        f"subjectAltName={san_string}",
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to generate self-signed certificate: {err.stderr}") from err
    except FileNotFoundError as err:
        raise RuntimeError("OpenSSL executable ('openssl') was not found in PATH.") from err

    return cert_path, key_path


def get_or_create_server_cert_and_key(
    cert_file: str | None = None,
    key_file: str | None = None,
    generate_self_signed: bool = False,
    hostname: str = "localhost",
) -> tuple[str, str]:
    """Resolve or generate certificate and private key file paths.

    Args:
        cert_file: Path to certificate PEM file.
        key_file: Path to private key PEM file.
        generate_self_signed: If True, generate self-signed certificate.
        hostname: Hostname for self-signed certificate SAN.

    Returns:
        Tuple of (cert_file, key_file).
    """
    if generate_self_signed:
        return generate_self_signed_cert(cert_path=cert_file, key_path=key_file, hostname=hostname)
    elif not cert_file or not key_file:
        raise ValueError("Either provide both cert_file and key_file, or set generate_self_signed=True")

    if not os.path.exists(cert_file):
        raise FileNotFoundError(f"Certificate file not found: {cert_file}")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Private key file not found: {key_file}")

    return cert_file, key_file


def create_server_ssl_context(
    cert_file: str,
    key_file: str,
) -> ssl.SSLContext:
    """Create and configure a server SSLContext from certificate and private key files.

    Args:
        cert_file: Path to certificate PEM file.
        key_file: Path to private key PEM file.

    Returns:
        Configured ssl.SSLContext for TLS server.
    """
    if not os.path.exists(cert_file):
        raise FileNotFoundError(f"Certificate file not found: {cert_file}")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Private key file not found: {key_file}")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Ensure modern TLS versions
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ssl_context


def create_grpc_server_credentials(
    cert_file: str,
    key_file: str,
) -> grpc.ServerCredentials:
    """Create gRPC server SSL credentials from certificate and private key files.

    Args:
        cert_file: Path to certificate PEM file.
        key_file: Path to private key PEM file.

    Returns:
        Configured grpc.ServerCredentials for TLS server.
    """
    if not os.path.exists(cert_file):
        raise FileNotFoundError(f"Certificate file not found: {cert_file}")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Private key file not found: {key_file}")

    with open(key_file, "rb") as f:
        private_key = f.read()
    with open(cert_file, "rb") as f:
        certificate_chain = f.read()

    return grpc.ssl_server_credentials(((private_key, certificate_chain),))
