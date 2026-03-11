from typing import TypedDict, List, Optional


class TextContent(TypedDict):
    odata_type: str  # @odata.type -> "microsoft.graph.textContent"
    data: str


class ProcessConversationMetadata(TypedDict):
    odata_type: str  # @odata.type -> "microsoft.graph.processConversationMetadata"
    identifier: str
    content: TextContent
    name: str
    correlationId: str
    sequenceNumber: int
    isTruncated: bool
    createdDateTime: str
    modifiedDateTime: str


class ActivityMetadata(TypedDict):
    activity: str  # e.g., "uploadText", "downloadText"


class OperatingSystemSpecifications(TypedDict):
    operatingSystemPlatform: str
    operatingSystemVersion: str


class DeviceMetadata(TypedDict, total=False):
    deviceType: str
    operatingSystemSpecifications: OperatingSystemSpecifications
    ipAddress: str


class PolicyLocationApplication(TypedDict):
    odata_type: str  # @odata.type -> "microsoft.graph.policyLocationApplication"
    value: str


class ProtectedApplicationMetadata(TypedDict, total=False):
    name: str
    version: str
    applicationLocation: PolicyLocationApplication


class IntegratedApplicationMetadata(TypedDict):
    name: str
    version: str


class ProcessContentRequest(TypedDict):
    contentEntries: List[ProcessConversationMetadata]
    activityMetadata: ActivityMetadata
    deviceMetadata: Optional[DeviceMetadata]
    protectedAppMetadata: Optional[ProtectedApplicationMetadata]
    integratedAppMetadata: IntegratedApplicationMetadata


class ProcessContentRequestBody(TypedDict):
    contentToProcess: ProcessContentRequest
