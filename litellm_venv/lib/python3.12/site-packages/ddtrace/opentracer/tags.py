from collections import namedtuple


TagNames = namedtuple(
    "TagNames",
    [
        "RESOURCE_NAME",
        "SAMPLING_PRIORITY",
        "SERVICE_NAME",
        "SPAN_TYPE",
        "TARGET_HOST",
        "TARGET_PORT",
    ],
)

Tags = TagNames(
    RESOURCE_NAME="resource.name",
    SAMPLING_PRIORITY="sampling.priority",
    SERVICE_NAME="service.name",
    TARGET_HOST="out.host",
    TARGET_PORT="network.destination.port",
    SPAN_TYPE="span.type",
)
