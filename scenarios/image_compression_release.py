#!/usr/bin/env python3
"""Compress a real image using litellm.extras.images.compress_image."""

from __future__ import annotations

import os
import sys

from dotenv import find_dotenv, load_dotenv

from litellm.extras.images import compress_image

load_dotenv(find_dotenv())


def run() -> None:
    image_path = os.getenv("RELEASE_IMAGE_PATH")
    if not image_path:
        print("RELEASE_IMAGE_PATH not set; skipping image compression scenario.")
        sys.exit(0)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    data_url = compress_image(image_path, max_kb=256)
    print(data_url[:120] + "...")


if __name__ == "__main__":
    run()
