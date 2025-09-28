#!/usr/bin/env python3
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import os
for key in sorted(os.environ):
    if key.startswith(('LITELLM', 'CODEX')):
        print(f"{key}={os.environ[key]}")
