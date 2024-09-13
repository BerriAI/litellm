class DatadogPayload(TypedDict, total=False):
    ddsource: str
    ddtags: str
    hostname: str
    message: str
    service: str
