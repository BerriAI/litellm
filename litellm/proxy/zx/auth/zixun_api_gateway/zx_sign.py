import hashlib
import uuid
from typing import Dict, Optional


class ZXSign:
    DEBUG = False

    @staticmethod
    def sha256_hex(data: str) -> str:
        """
        Generate SHA256 signature and convert to uppercase

        :param data: String to be signed
        :return: Uppercase SHA256 signature
        """
        return hashlib.sha256(data.encode("utf-8")).hexdigest().upper()

    @staticmethod
    def get_uuid() -> str:
        """
        Generate UUID without hyphens

        :return: Uppercase UUID without hyphens
        """
        return str(uuid.uuid4()).replace("-", "").upper()

    @staticmethod
    def get_sign_header(
        app_id: str, app_key: str, trace_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate headers for API gateway call

        :param app_id: Assigned channel number
        :param app_key: Assigned channel secret
        :param trace_id: Optional trace ID for request tracking
        :return: Dictionary of signed headers
        """
        import time

        # Use current timestamp in milliseconds
        ts = int(time.time() * 1000)

        # Use provided trace_id or generate a new one
        if trace_id is None:
            trace_id = ZXSign.get_uuid()

        # Generate nonce (first 6 characters of UUID)
        nonce = ZXSign.get_uuid()[:6]

        # Create original signature string
        ori_sign_str = f"{app_id}{trace_id}{ts}{nonce}{app_key}"

        # Generate signature
        sign = ZXSign.sha256_hex(ori_sign_str)

        # Create headers dictionary
        headers = {
            "ts": str(ts),
            "traceId": trace_id,
            "noce": nonce,
            "sign": sign,
            "appId": app_id,
        }

        # Debug logging
        if ZXSign.DEBUG:
            print(f"headers: {headers}")
            print(f"oriSignStr: {ori_sign_str}")
            print(f"sign: {sign}")

        return headers


# Optional: Example usage
if __name__ == "__main__":
    # Enable debug mode
    ZXSign.DEBUG = True

    # Example call
    headers = ZXSign.get_sign_header(app_id="your_app_id", app_key="your_app_key")
