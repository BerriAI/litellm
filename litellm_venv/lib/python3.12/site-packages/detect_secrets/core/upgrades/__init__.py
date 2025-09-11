"""
All modules in this package needs to have the following boilerplate:

```python
from typing import Any
from typing import Dict


def upgrade(baseline: Dict[str, Any]) -> None:
    pass
```

These upgrades SHOULD NOT be used to add new plugins, as that will require more information
than can be obtained from the baseline itself.
"""
