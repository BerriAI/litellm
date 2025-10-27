from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict


class BedrockKBLocation(TypedDict, total=False):
    """Location information for a retrieved document."""

    type: str
    s3Location: Optional[dict]
    webLocation: Optional[dict]
    kendraDocumentLocation: Optional[dict]
    salesforceLocation: Optional[dict]
    sharePointLocation: Optional[dict]
    confluenceLocation: Optional[dict]
    customDocumentLocation: Optional[dict]
    sqlLocation: Optional[dict]


class BedrockKBRowValue(TypedDict):
    """Row value in a retrieved document."""

    columnName: str
    columnValue: str
    type: str


class BedrockKBContent(TypedDict, total=False):
    """Content of a retrieved document."""

    type: str
    text: Optional[str]
    byteContent: Optional[str]
    row: Optional[List[BedrockKBRowValue]]


class BedrockKBRetrievalResult(TypedDict, total=False):
    """Individual result from a knowledge base retrieval."""

    content: Optional[BedrockKBContent]
    location: Optional[BedrockKBLocation]
    score: Optional[float]
    metadata: Optional[Dict[str, Any]]


class BedrockKBResponse(TypedDict, total=False):
    """Response from a Bedrock Knowledge Base retrieval request."""

    guardrailAction: Optional[Literal["INTERVENED", "NONE"]]
    nextToken: Optional[str]
    retrievalResults: Optional[List[BedrockKBRetrievalResult]]


################ Bedrock Knowledge Base Request Types #################
#########################################################################
#########################################################################


class BedrockKBMetadataAttribute(TypedDict, total=False):
    """Metadata attribute configuration for implicit filtering."""

    description: Optional[str]
    key: Optional[str]
    type: Optional[str]


class BedrockKBImplicitFilterConfiguration(TypedDict, total=False):
    """Configuration for implicit filtering."""

    metadataAttributes: Optional[List[BedrockKBMetadataAttribute]]
    modelArn: Optional[str]


class BedrockKBSelectiveModeConfiguration(TypedDict, total=False):
    """Configuration for selective mode in reranking."""

    pass  # This can be expanded based on actual requirements


class BedrockKBMetadataConfiguration(TypedDict, total=False):
    """Metadata configuration for reranking."""

    selectionMode: Optional[str]
    selectiveModeConfiguration: Optional[BedrockKBSelectiveModeConfiguration]


class BedrockKBModelConfiguration(TypedDict, total=False):
    """Model configuration for reranking."""

    additionalModelRequestFields: Optional[Dict[str, Any]]
    modelArn: Optional[str]


class BedrockKBRerankingConfiguration(TypedDict, total=False):
    """Configuration for reranking in vector search."""

    bedrockRerankingConfiguration: Optional[
        Dict[str, Any]
    ]  # This could be further typed if needed
    type: Optional[str]


class BedrockKBVectorSearchConfiguration(TypedDict, total=False):
    """Configuration for vector search."""

    filter: Optional[Dict[str, Any]]
    implicitFilterConfiguration: Optional[BedrockKBImplicitFilterConfiguration]
    numberOfResults: Optional[int]
    overrideSearchType: Optional[str]
    rerankingConfiguration: Optional[BedrockKBRerankingConfiguration]


class BedrockKBRetrievalConfiguration(TypedDict, total=False):
    """Configuration for retrieval."""

    vectorSearchConfiguration: Optional[BedrockKBVectorSearchConfiguration]


class BedrockKBRetrievalQuery(TypedDict, total=False):
    """Query structure for retrieval."""

    text: Optional[str]


class BedrockKBGuardrailConfiguration(TypedDict, total=False):
    """Configuration for guardrails."""

    guardrailId: Optional[str]
    guardrailVersion: Optional[str]


class BedrockKBRequest(TypedDict, total=False):
    """Complete request structure for Bedrock Knowledge Base retrieval."""

    guardrailConfiguration: Optional[BedrockKBGuardrailConfiguration]
    nextToken: Optional[str]
    retrievalConfiguration: Optional[BedrockKBRetrievalConfiguration]
    retrievalQuery: BedrockKBRetrievalQuery


#########################################################################
#########################################################################
#########################################################################
