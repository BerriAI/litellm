"""Category and policy-template YAML for the content filter guardrail.

Kept out of ``guardrail_hooks/litellm_content_filter/`` so the packaged paths
stay under the Windows MAX_PATH budget enforced by
``tests/windows_tests/check_windows_wheel_install.py``. That package directory
stays a search root so files a deployment copied there before the move keep
loading.
"""

import itertools
import os
from typing import Final

from litellm.proxy.common_utils.path_utils import try_safe_join

DATA_DIR: Final = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_DIR: Final = os.path.join(DATA_DIR, "categories")
POLICY_TEMPLATES_DIR: Final = os.path.join(DATA_DIR, "policy_templates")
LEGACY_DATA_DIR: Final = os.path.join(os.path.dirname(DATA_DIR), "guardrail_hooks", "litellm_content_filter")
DATA_ROOTS: Final = (DATA_DIR, LEGACY_DATA_DIR)


def category_dirs(roots: tuple[str, ...] = DATA_ROOTS) -> tuple[str, ...]:
    """Every ``categories/`` folder that exists under the roots, bundled first."""
    return tuple(d for d in (os.path.join(root, "categories") for root in roots) if os.path.isdir(d))


def find_category_file(category_name: str, roots: tuple[str, ...] = DATA_ROOTS) -> str | None:
    """First ``<name>.yaml`` or ``<name>.json`` across the category folders, or None.

    A name that would escape its folder (``../x``) never matches.
    """
    candidates: Final = (
        try_safe_join(d, f"{category_name}{ext}")
        for d, ext in itertools.product(category_dirs(roots), (".yaml", ".json"))
    )
    return next((c for c in candidates if c is not None and os.path.isfile(c)), None)
