# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Google AI Python SDK

## Setup

```posix-terminal
pip install google-generativeai
```

## GenerativeModel

Use `genai.GenerativeModel` to access the API:

```
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['API_KEY'])

model = genai.GenerativeModel(name='gemini-pro')
response = model.generate_content('Please summarise this document: ...')

print(response.text)
```

See the [python quickstart](https://ai.google.dev/tutorials/python_quickstart) for more details.
"""
from __future__ import annotations

from google.generativeai import version

from google.generativeai import types
from google.generativeai.types import GenerationConfig


from google.generativeai.discuss import chat
from google.generativeai.discuss import chat_async
from google.generativeai.discuss import count_message_tokens

from google.generativeai.embedding import embed_content
from google.generativeai.embedding import embed_content_async

from google.generativeai.files import upload_file
from google.generativeai.files import get_file
from google.generativeai.files import list_files
from google.generativeai.files import delete_file

from google.generativeai.generative_models import GenerativeModel
from google.generativeai.generative_models import ChatSession

from google.generativeai.text import generate_text
from google.generativeai.text import generate_embeddings
from google.generativeai.text import count_text_tokens

from google.generativeai.models import list_models
from google.generativeai.models import list_tuned_models

from google.generativeai.models import get_model
from google.generativeai.models import get_base_model
from google.generativeai.models import get_tuned_model

from google.generativeai.models import create_tuned_model
from google.generativeai.models import update_tuned_model
from google.generativeai.models import delete_tuned_model

from google.generativeai.operations import list_operations
from google.generativeai.operations import get_operation


from google.generativeai.client import configure

__version__ = version.__version__

del discuss
del embedding
del files
del generative_models
del text
del models
del client
del operations
del version
