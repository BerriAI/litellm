import datetime
import ipaddress
import ssl
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def write_self_signed_cert(cert_dir: Path, names: tuple[str, ...] = ("localhost",)) -> tuple[Path, Path]:
    """Write a loopback certificate valid for `names` and 127.0.0.1; returns (cert path, key path)."""
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    subject: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])])
    alternatives: Final[tuple[x509.GeneralName, ...]] = tuple(x509.DNSName(name) for name in names) + (
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    )
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=7))
        .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_file: Final = cert_dir / "cert.pem"
    key_file: Final = cert_dir / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


def server_context(cert_file: Path, key_file: Path) -> ssl.SSLContext:
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return context
