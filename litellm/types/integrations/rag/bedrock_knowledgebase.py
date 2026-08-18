from typing import Any, Literal

from typing_extensions import TypedDict


class BedrockKBLocation(TypedDict, total=False):
    """Location information for a retrieved document."""

    type: str
    s3Location: dict | None
    webLocation: dict | None
    kendraDocumentLocation: dict | None
    salesforceLocation: dict | None
    sharePointLocation: dict | None
    confluenceLocation: dict | None
    customDocumentLocation: dict | None
    sqlLocation: dict | None


class BedrockKBRowValue(TypedDict):
    """Row value in a retrieved document."""

    columnName: str
    columnValue: str
    type: str


class BedrockKBContent(TypedDict, total=False):
    """Content of a retrieved document."""

    type: str
    text: str | None
    byteContent: str | None
    row: list[BedrockKBRowValue] | None


class BedrockKBRetrievalResult(TypedDict, total=False):
    """Individual result from a knowledge base retrieval."""

    content: BedrockKBContent | None
    location: BedrockKBLocation | None
    score: float | None
    metadata: dict[str, Any] | None


class BedrockKBResponse(TypedDict, total=False):
    """Response from a Bedrock Knowledge Base retrieval request."""

    guardrailAction: Literal["INTERVENED", "NONE"] | None
    nextToken: str | None
    retrievalResults: list[BedrockKBRetrievalResult] | None


################ Bedrock Knowledge Base Request Types #################
#########################################################################
#########################################################################


class BedrockKBMetadataAttribute(TypedDict, total=False):
    """Metadata attribute configuration for implicit filtering."""

    description: str | None
    key: str | None
    type: str | None


class BedrockKBImplicitFilterConfiguration(TypedDict, total=False):
    """Configuration for implicit filtering."""

    metadataAttributes: list[BedrockKBMetadataAttribute] | None
    modelArn: str | None


class BedrockKBSelectiveModeConfiguration(TypedDict, total=False):
    """Configuration for selective mode in reranking."""

    # This can be expanded based on actual requirements


class BedrockKBMetadataConfiguration(TypedDict, total=False):
    """Metadata configuration for reranking."""

    selectionMode: str | None
    selectiveModeConfiguration: BedrockKBSelectiveModeConfiguration | None


class BedrockKBModelConfiguration(TypedDict, total=False):
    """Model configuration for reranking."""

    additionalModelRequestFields: dict[str, Any] | None
    modelArn: str | None


class BedrockKBRerankingConfiguration(TypedDict, total=False):
    """Configuration for reranking in vector search."""

    bedrockRerankingConfiguration: dict[str, Any] | None  # This could be further typed if needed
    type: str | None


class BedrockKBVectorSearchConfiguration(TypedDict, total=False):
    """Configuration for vector search."""

    filter: dict[str, Any] | None
    implicitFilterConfiguration: BedrockKBImplicitFilterConfiguration | None
    numberOfResults: int | None
    overrideSearchType: str | None
    rerankingConfiguration: BedrockKBRerankingConfiguration | None


class BedrockKBRetrievalConfiguration(TypedDict, total=False):
    """Configuration for retrieval."""

    vectorSearchConfiguration: BedrockKBVectorSearchConfiguration | None


class BedrockKBRetrievalQuery(TypedDict, total=False):
    """Query structure for retrieval."""

    text: str | None


class BedrockKBGuardrailConfiguration(TypedDict, total=False):
    """Configuration for guardrails."""

    guardrailId: str | None
    guardrailVersion: str | None


class BedrockKBRequest(TypedDict, total=False):
    """Complete request structure for Bedrock Knowledge Base retrieval."""

    guardrailConfiguration: BedrockKBGuardrailConfiguration | None
    nextToken: str | None
    retrievalConfiguration: BedrockKBRetrievalConfiguration | None
    retrievalQuery: BedrockKBRetrievalQuery


#########################################################################
#########################################################################
#########################################################################
