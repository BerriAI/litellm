class AttributeForwarder:
    def __init__(self, target: object) -> None:
        self._target = target

    def __getattr__(self, name: str) -> object:
        return getattr(self._target, name)
