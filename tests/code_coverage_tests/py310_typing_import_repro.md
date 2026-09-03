# Python 3.10 typing import regression

The published 1.98.0 package fails to import on Python 3.10 because `compact.py` imports `NotRequired` from `typing`

```text
$ cd /home/ubuntu/py310-repro
$ /usr/bin/python3.10 -m venv .venv && .venv/bin/pip install -q litellm==1.98.0 && .venv/bin/python -c "import litellm"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/__init__.py", line 1362, in <module>
    from .llms.anthropic.experimental_pass_through.messages.handler import *
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/messages/handler.py", line 34, in <module>
    from ..adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/adapters/__init__.py", line 1, in <module>
    from .transformation import LiteLLMAnthropicMessagesAdapter
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py", line 72, in <module>
    from litellm.llms.anthropic.experimental_pass_through.context_management import (
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/context_management/__init__.py", line 2, in <module>
    from .dispatcher import apply_context_management
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/context_management/dispatcher.py", line 11, in <module>
    from .editors import apply_clear_tool_uses_20250919, apply_compact_20260112
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/context_management/editors/__init__.py", line 2, in <module>
    from .compact import apply_compact_20260112
  File "/home/ubuntu/py310-repro/.venv/lib/python3.10/site-packages/litellm/llms/anthropic/experimental_pass_through/context_management/editors/compact.py", line 17, in <module>
    from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, Optional, TypedDict, Union, cast
ImportError: cannot import name 'NotRequired' from 'typing' (/usr/lib/python3.10/typing.py)
```

The staging checkout imports `NotRequired` from `typing_extensions`, so importing the checkout succeeds. The checkout
currently does not expose `litellm.__version__`, and its default install does not include the proxy dependency required
by the `litellm` CLI

```text
$ .venv/bin/pip install -q -e /home/ubuntu/repos/litellm
$ .venv/bin/python -c "import litellm; print(litellm.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/ubuntu/repos/litellm/litellm/__init__.py", line 2392, in __getattr__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
AttributeError: module 'litellm' has no attribute '__version__'

$ .venv/bin/litellm --version
Traceback (most recent call last):
  File "/home/ubuntu/repos/litellm/litellm/proxy/proxy_cli.py", line 1011, in run_server
    from .proxy_server import (
  File "/home/ubuntu/repos/litellm/litellm/proxy/proxy_server.py", line 40, in <module>
    import websockets
ModuleNotFoundError: No module named 'websockets'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ubuntu/py310-repro/.venv/bin/litellm", line 8, in <module>
    sys.exit(run_server())
  File "/home/ubuntu/repos/litellm/litellm/proxy/proxy_cli.py", line 1018, in run_server
    raise ModuleNotFoundError(f"Missing dependency {e}. Run `pip install 'litellm[proxy]'`")
ModuleNotFoundError: Missing dependency No module named 'websockets'. Run `pip install 'litellm[proxy]'`
ModuleNotFoundError: Missing dependency No module named 'websockets'. Run `pip install 'litellm[proxy]'`
```

The import itself succeeds with the checkout overlay:

```text
$ .venv/bin/python -c "import litellm; print('import ok')"
import ok
```
