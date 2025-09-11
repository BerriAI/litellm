from typing import Any  # noqa:F401


class AttrDict(dict):
    """
    dict implementation that allows for item attribute access


    Example::

       data = AttrDict()
       data['key'] = 'value'
       print(data['key'])

       data.key = 'new-value'
       print(data.key)

       # Convert an existing `dict`
       data = AttrDict(dict(key='value'))
       print(data.key)
    """

    def __getattr__(self, key):
        # type: (str) -> Any
        if key in self:
            return self[key]
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        # type: (str, Any) -> None
        # 1) Ensure if the key exists from a dict key we always prefer that
        # 2) If we do not have an existing key but we do have an attr, set that
        # 3) No existing key or attr exists, so set a key
        if key in self:
            # Update any existing key
            self[key] = value
        elif hasattr(self, key):
            # Allow overwriting an existing attribute, e.g. `self.global_config = dict()`
            object.__setattr__(self, key, value)
        else:
            # Set a new key
            self[key] = value
