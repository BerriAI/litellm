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
"""A high level client library for generative AI.

## Setup

```posix-terminal
pip install google-generativeai
```

```
import google.generativeai as palm
import os

palm.configure(api_key=os.environ['API_KEY'])
```

## Text

Use the `palm.generate_text` function to have the model complete some initial
text.

```
response = palm.generate_text(prompt="The opposite of hot is")
print(response.result) #  'cold.'
```

## Chat

Use the `palm.chat` function to have a discussion with a model:

```
response = palm.chat(messages=["Hello."])
print(response.last) #  'Hello! What can I help you with?'
response.reply("Can you tell me a joke?")
```

## Models

Use the model service discover models and find out more about them:

Use `palm.get_model` to get details if you know a model's name:

```
model = palm.get_model('models/chat-bison-001') # 🦬
```

Use `palm.list_models` to discover models:

```
import pprint
for model in palm.list_models():
    pprint.pprint(model) # 🦎🦦🦬🦄
```

"""
from __future__ import annotations

from google.generativeai import types
from google.generativeai import version

from google.generativeai.discuss import chat
from google.generativeai.discuss import chat_async
from google.generativeai.discuss import count_message_tokens

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
del text
del models
del client
del version
