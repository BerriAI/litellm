"""Category and policy-template YAML for the content filter guardrail.

Kept out of ``guardrail_hooks/litellm_content_filter/`` so the packaged paths
stay under the Windows MAX_PATH budget enforced by
``tests/windows_tests/check_windows_wheel_install.py``.
"""

import os
from typing import Final

DATA_DIR: Final = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_DIR: Final = os.path.join(DATA_DIR, "categories")
POLICY_TEMPLATES_DIR: Final = os.path.join(DATA_DIR, "policy_templates")
