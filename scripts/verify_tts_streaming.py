#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
import time
from typing import Optional

import httpx


def build_url(base_url: str, endpoint_path: str) -> str:
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    return base_url + endpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TTS streaming via chunked transfer")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://0.0.0.0:4000"),
        help="Base URL for the API (default from OPENAI_BASE_URL or http://0.0.0.0:4000)",
    )
    parser.add_argument(
        "--endpoint-path",
        default="/v1/audio/speech",
        help="Endpoint path to call (e.g. /v1/audio/speech or /openai/audio/speech)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-tts",
        help="Model name (default: gpt-4o-mini-tts)",
    )
    parser.add_argument(
        "--voice",
        default="shimmer",
        help="Voice to use (default: shimmer)",
    )
    parser.add_argument(
        "--input",
        default=(
            "Once upon a time, in a bustling city nestled between rolling hills and a sparkling river, there lived a young inventor named Elara. Elara was known throughout the city for her boundless curiosity and her knack for creating marvelous contraptions from the most ordinary of objects. One day, while exploring the attic of her late grandfather’s house, she stumbled upon a dusty, leather-bound journal filled with cryptic notes and intricate sketches of a mysterious machine. Intrigued, Elara spent days deciphering the journal, piecing together the purpose of the device. It was said to be a portal, capable of bridging worlds and connecting distant realms. Driven by excitement and a sense of adventure, Elara gathered the necessary parts—cogs, wires, crystals, and a peculiar brass key—and began assembling the machine in her workshop. As she tightened the final bolt and inserted the key, the device hummed to life, casting a shimmering blue light across the room. With a deep breath, Elara stepped forward and activated the portal. Instantly, she was enveloped in a whirlwind of colors and sounds, feeling herself transported beyond the boundaries of her world. When the light faded, she found herself standing in a lush, enchanted forest, where trees whispered secrets and fantastical creatures roamed freely. Elara realized she had crossed into a realm of endless possibilities, where her inventions could shape the very fabric of reality. Determined to explore and learn, she set off down a winding path, eager to uncover the wonders and challenges that awaited her in this extraordinary new world. And so began Elara’s greatest adventure, one that would test her ingenuity, courage, and heart, and ultimately reveal the true power of imagination and discovery."
        ),
        help="Text to synthesize",
    )
    parser.add_argument(
        "--response-format",
        default="mp3",
        help="Audio response format (default: mp3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write audio to (if omitted, data is discarded)",
    )
    parser.add_argument(
        "--http2",
        action="store_true",
        help="Enable HTTP/2 (default: off). Leave off to see chunked headers in HTTP/1.1",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in the environment", file=sys.stderr)
        return 2

    url = build_url(args.base_url, args.endpoint_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    json_body = {
        "model": args.model,
        "input": args.input,
        "voice": args.voice,
        "response_format": args.response_format,
    }

    print(f"Requesting: {url}")
    print(f"HTTP/2: {'on' if args.http2 else 'off'} (HTTP/1.1 if off)")

    # Force HTTP/1.1 by default to make Transfer-Encoding: chunked visible when streaming.
    # For HTTP/2, chunked header will not be present even when streaming works.
    start_req = time.time()
    first_byte_at: Optional[float] = None
    total_bytes = 0

    with httpx.Client(http2=args.http2, timeout=None) as client:
        with client.stream("POST", url, headers=headers, json=json_body) as resp:
            status = resp.status_code
            # Print key headers that indicate buffering vs streaming
            cl = resp.headers.get("content-length")
            te = resp.headers.get("transfer-encoding")
            server = resp.headers.get("server")
            print(f"Status: {status}")
            print(f"Content-Type: {resp.headers.get('content-type')}")
            print(f"Content-Length: {cl}")
            print(f"Transfer-Encoding: {te}")
            print(f"Server: {server}")

            # Stream body
            sink_cm = open(args.output, "wb") if args.output else contextlib.nullcontext()
            with sink_cm as sink:
                for chunk in resp.iter_bytes():
                    if not first_byte_at:
                        first_byte_at = time.time()
                        print(
                            f"First byte after {first_byte_at - start_req:.3f}s"
                        )
                    total_bytes += len(chunk)
                    if sink and hasattr(sink, "write"):
                        sink.write(chunk)  # type: ignore

    end = time.time()
    print(f"Total bytes: {total_bytes}")
    print(f"Total time: {end - start_req:.3f}s")
    if first_byte_at:
        print(f"Time to first byte: {first_byte_at - start_req:.3f}s")

    print()
    print("Interpretation:")
    print("- If Content-Length is absent and Transfer-Encoding is chunked (HTTP/1.1), it streamed.")
    print("- If Content-Length is present, the response was buffered by an intermediary or origin.")
    print("- Even with HTTP/2 (no chunked header), early first byte indicates streaming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


