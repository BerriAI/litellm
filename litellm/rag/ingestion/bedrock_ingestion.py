"""
Bedrock-specific RAG Ingestion implementation.

Bedrock Knowledge Bases handle embedding internally when files are ingested,
so this implementation uploads files to S3 and triggers ingestion jobs.

Supports two modes:
1. Use existing KB: Provide vector_store_id (KB ID)
2. Auto-create KB: Don't provide vector_store_id - creates all AWS resources automatically
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


def _get_str_or_none(value: Any) -> str | None:
    """Cast config value to Optional[str]."""
    return str(value) if value is not None else None


def _get_int(value: Any, default: int) -> int:
    """Cast config value to int with default."""
    if value is None:
        return default
    return int(value)


def _normalize_principal_arn(caller_arn: str, account_id: str) -> str:
    """
    Normalize a caller ARN to the format required by OpenSearch data access policies.

    OpenSearch Serverless data access policies require:
    - IAM users: arn:aws:iam::account-id:user/user-name
    - IAM roles: arn:aws:iam::account-id:role/role-name

    But get_caller_identity() returns for assumed roles:
    - arn:aws:sts::account-id:assumed-role/role-name/session-name

    This function converts assumed-role ARNs to the proper IAM role ARN format.
    """
    if ":assumed-role/" in caller_arn:
        # Extract role name from assumed-role ARN
        # Format: arn:aws:sts::ACCOUNT:assumed-role/ROLE-NAME/SESSION-NAME
        parts: Final = caller_arn.split("/")
        if len(parts) >= 2:
            role_name: Final = parts[1]
            return f"arn:aws:iam::{account_id}:role/{role_name}"
    return caller_arn


class BedrockRAGIngestion(BaseRAGIngestion, BaseAWSLLM):
    """
    Bedrock Knowledge Base RAG ingestion.

    Supports two modes:
    1. **Use existing KB**: Provide vector_store_id
    2. **Auto-create KB**: Don't provide vector_store_id - creates S3 bucket,
       OpenSearch Serverless collection, IAM role, KB, and data source automatically

    Optional config:
    - vector_store_id: Existing KB ID (if not provided, auto-creates)
    - s3_bucket: S3 bucket (auto-created if not provided)
    - embedding_model: Bedrock embedding model (default: amazon.titan-embed-text-v2:0)
    - wait_for_ingestion: Wait for completion (default: True)
    - ingestion_timeout: Max seconds to wait (default: 300)

    AWS Auth (uses BaseAWSLLM):
    - aws_access_key_id, aws_secret_access_key, aws_session_token
    - aws_region_name (default: us-west-2)
    - aws_role_name, aws_session_name, aws_profile_name
    - aws_web_identity_token, aws_sts_endpoint, aws_external_id
    """

    def __init__(
        self,
        ingest_options: RAGIngestOptions,
        router: Router | None = None,
    ):
        BaseRAGIngestion.__init__(self, ingest_options=ingest_options, router=router)
        BaseAWSLLM.__init__(self)

        # Use vector_store_id as unified param (maps to knowledge_base_id)
        self.knowledge_base_id = self.vector_store_config.get("vector_store_id") or self.vector_store_config.get(
            "knowledge_base_id"
        )

        # Optional config
        self._data_source_id = self.vector_store_config.get("data_source_id")
        self._s3_bucket = self.vector_store_config.get("s3_bucket")
        self._s3_prefix: str | None = (
            str(self.vector_store_config.get("s3_prefix")) if self.vector_store_config.get("s3_prefix") else None
        )
        self.embedding_model = self.vector_store_config.get("embedding_model") or "amazon.titan-embed-text-v2:0"

        self.wait_for_ingestion = self.vector_store_config.get("wait_for_ingestion", False)
        self.ingestion_timeout: int = _get_int(self.vector_store_config.get("ingestion_timeout"), 300)

        # Get AWS region using BaseAWSLLM method
        _aws_region: Final = self.vector_store_config.get("aws_region_name")
        self.aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=str(_aws_region) if _aws_region else None
        )

        # Will be set during initialization
        self.data_source_id: str | None = None
        self.s3_bucket: str | None = None
        self.s3_prefix: str = self._s3_prefix or "data/"
        self._config_initialized = False

        # Track resources we create (for cleanup if needed)
        self._created_resources: dict[str, Any] = {}

    async def _ensure_config_initialized(self):
        """Lazily initialize KB config - either detect from existing or create new."""
        if self._config_initialized:
            return

        if self.knowledge_base_id:
            # Use existing KB - auto-detect data source and S3 bucket
            self._auto_detect_config()
        else:
            # No KB provided - create everything from scratch
            await self._create_knowledge_base_infrastructure()

        self._config_initialized = True

    def _auto_detect_config(self):
        """Auto-detect data source ID and S3 bucket from existing Knowledge Base."""
        verbose_logger.debug("Auto-detecting data source and S3 bucket for KB=%s", self.knowledge_base_id)

        bedrock_agent: Final = self._get_boto3_client("bedrock-agent")

        # List data sources for this KB
        ds_response: Final = bedrock_agent.list_data_sources(knowledgeBaseId=self.knowledge_base_id)
        data_sources: Final = ds_response.get("dataSourceSummaries", [])

        if not data_sources:
            raise ValueError(
                f"No data sources found for Knowledge Base {self.knowledge_base_id}. "
                "Please create a data source first or provide data_source_id and s3_bucket."
            )

        # Use first data source (or user-provided override)
        if self._data_source_id:
            self.data_source_id = self._data_source_id
        else:
            self.data_source_id = data_sources[0]["dataSourceId"]
            verbose_logger.info("Auto-detected data source: %s", self.data_source_id)

        # Get data source details for S3 bucket
        ds_details: Final = bedrock_agent.get_data_source(
            knowledgeBaseId=self.knowledge_base_id,
            dataSourceId=self.data_source_id,
        )

        s3_config = ds_details.get("dataSource", {}).get("dataSourceConfiguration", {}).get("s3Configuration", {})

        bucket_arn: Final = s3_config.get("bucketArn", "")
        if bucket_arn:
            # Extract bucket name from ARN: arn:aws:s3:::bucket-name
            self.s3_bucket = self._s3_bucket or bucket_arn.split(":")[-1]
            verbose_logger.info("Auto-detected S3 bucket: %s", self.s3_bucket)

            # Use inclusion prefix if available
            prefixes: Final = s3_config.get("inclusionPrefixes", [])
            if prefixes and not self._s3_prefix:
                self.s3_prefix = prefixes[0]
        else:
            if not self._s3_bucket:
                raise ValueError(
                    f"Could not auto-detect S3 bucket for data source {self.data_source_id}. "
                    "Please provide s3_bucket in config."
                )
            self.s3_bucket = self._s3_bucket

    async def _create_knowledge_base_infrastructure(self):
        """Create all AWS resources needed for a new Knowledge Base."""
        verbose_logger.info("Creating new Bedrock Knowledge Base infrastructure...")

        # Generate unique names
        unique_id: Final = uuid.uuid4().hex[:8]
        kb_name: Final = self.ingest_name or f"litellm-kb-{unique_id}"

        # Get AWS account ID and caller ARN (for data access policy)
        sts: Final = self._get_boto3_client("sts")
        caller_identity: Final = sts.get_caller_identity()
        account_id: Final = caller_identity["Account"]
        caller_arn: Final = caller_identity["Arn"]

        # Step 1: Create S3 bucket (if not provided)
        self.s3_bucket = self._s3_bucket or self._create_s3_bucket(unique_id)

        # Step 2: Create OpenSearch Serverless collection
        collection_name, collection_arn = await self._create_opensearch_collection(unique_id, account_id, caller_arn)

        # Step 3: Create OpenSearch index
        await self._create_opensearch_index(collection_name)

        # Step 4: Create IAM role for Bedrock
        role_arn: Final = await self._create_bedrock_role(unique_id, account_id, collection_arn)

        # Step 5: Create Knowledge Base
        self.knowledge_base_id = await self._create_knowledge_base(kb_name, role_arn, collection_arn)

        # Step 6: Create Data Source
        self.data_source_id = self._create_data_source(kb_name)

        verbose_logger.info(
            "Created KB infrastructure: kb_id=%s, ds_id=%s, bucket=%s",
            self.knowledge_base_id,
            self.data_source_id,
            self.s3_bucket,
        )

    def _create_s3_bucket(self, unique_id: str) -> str:
        """Create S3 bucket for KB data source."""
        s3: Final = self._get_boto3_client("s3")
        bucket_name: Final = f"litellm-kb-{unique_id}"

        verbose_logger.debug("Creating S3 bucket: %s", bucket_name)

        create_params: Final[dict[str, Any]] = {"Bucket": bucket_name}
        if self.aws_region_name != "us-east-1":
            create_params["CreateBucketConfiguration"] = {"LocationConstraint": self.aws_region_name}

        s3.create_bucket(**create_params)
        self._created_resources["s3_bucket"] = bucket_name

        verbose_logger.info("Created S3 bucket: %s", bucket_name)
        return bucket_name

    async def _create_opensearch_collection(self, unique_id: str, account_id: str, caller_arn: str) -> tuple[str, str]:
        """Create OpenSearch Serverless collection for vector storage."""
        oss: Final = self._get_boto3_client("opensearchserverless")
        collection_name: Final = f"litellm-kb-{unique_id}"

        verbose_logger.debug("Creating OpenSearch Serverless collection: %s", collection_name)

        # Create encryption policy
        oss.create_security_policy(
            name=f"{collection_name}-enc",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Create network policy (public access for simplicity)
        oss.create_security_policy(
            name=f"{collection_name}-net",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection_name}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        # Create data access policy - include both root and actual caller ARN
        # This ensures the credentials being used have access to the collection
        # Normalize the caller ARN (convert assumed-role ARN to IAM role ARN if needed)
        normalized_caller_arn: Final = _normalize_principal_arn(caller_arn, account_id)
        verbose_logger.debug("Caller ARN: %s, Normalized: %s", caller_arn, normalized_caller_arn)

        principals = [f"arn:aws:iam::{account_id}:root", normalized_caller_arn]
        # Deduplicate in case caller is root
        principals = list(set(principals))

        oss.create_access_policy(
            name=f"{collection_name}-access",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": ["aoss:*"],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": ["aoss:*"],
                            },
                        ],
                        "Principal": principals,
                    }
                ]
            ),
        )

        # Create collection
        response: Final = oss.create_collection(
            name=collection_name,
            type="VECTORSEARCH",
        )
        collection_id: Final = response["createCollectionDetail"]["id"]
        self._created_resources["opensearch_collection"] = collection_name

        # Wait for collection to be active (use asyncio.sleep to avoid blocking)
        verbose_logger.debug("Waiting for OpenSearch collection to be active...")
        for _ in range(60):  # 5 min timeout
            status_response = oss.batch_get_collection(ids=[collection_id])
            status = status_response["collectionDetails"][0]["status"]
            if status == "ACTIVE":
                break
            await asyncio.sleep(5)
        else:
            raise TimeoutError("OpenSearch collection did not become active in time")

        collection_arn: Final = status_response["collectionDetails"][0]["arn"]
        verbose_logger.info("Created OpenSearch collection: %s", collection_name)

        # Wait for data access policy to propagate before returning
        # AWS recommends waiting 60+ seconds for policy propagation
        verbose_logger.debug("Waiting for data access policy to propagate (60s)...")
        await asyncio.sleep(60)

        return collection_name, collection_arn

    async def _create_opensearch_index(self, collection_name: str):
        """Create vector index in OpenSearch collection with retry logic."""
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth

        # Get credentials for signing
        credentials: Final = self.get_credentials(
            aws_access_key_id=_get_str_or_none(self.vector_store_config.get("aws_access_key_id")),
            aws_secret_access_key=_get_str_or_none(self.vector_store_config.get("aws_secret_access_key")),
            aws_session_token=_get_str_or_none(self.vector_store_config.get("aws_session_token")),
            aws_region_name=self.aws_region_name,
        )

        # Get collection endpoint
        oss: Final = self._get_boto3_client("opensearchserverless")
        collections: Final = oss.batch_get_collection(names=[collection_name])
        endpoint: Final = collections["collectionDetails"][0]["collectionEndpoint"]
        host: Final = endpoint.replace("https://", "")

        auth: Final = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.aws_region_name,
            "aoss",
            session_token=credentials.token,
        )

        client: Final = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        index_name: Final = "bedrock-kb-index"
        index_body: Final = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
            "mappings": {
                "properties": {
                    "bedrock-knowledge-base-default-vector": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "engine": "faiss",
                            "name": "hnsw",
                            "space_type": "l2",
                        },
                    },
                    "AMAZON_BEDROCK_METADATA": {"type": "text", "index": False},
                    "AMAZON_BEDROCK_TEXT_CHUNK": {"type": "text"},
                }
            },
        }

        # Retry logic for index creation - data access policy may take time to propagate
        max_retries: Final = 8
        retry_delay: Final = 20  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                client.indices.create(index=index_name, body=index_body)
                verbose_logger.info("Created OpenSearch index: %s", index_name)
                return
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "authorization_exception" in error_str.lower() or "security_exception" in error_str.lower():
                    verbose_logger.warning(
                        "OpenSearch index creation attempt %s/%s failed due to authorization. Waiting %ss for policy propagation...",
                        attempt + 1,
                        max_retries,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    # Non-auth error, raise immediately
                    raise

        # All retries exhausted
        raise RuntimeError(
            f"Failed to create OpenSearch index after {max_retries} attempts. "
            f"Data access policy may not have propagated. Last error: {last_error}"
        )

    async def _create_bedrock_role(self, unique_id: str, account_id: str, collection_arn: str) -> str:
        """Create IAM role for Bedrock KB."""
        iam: Final = self._get_boto3_client("iam")
        role_name: Final = f"litellm-bedrock-kb-{unique_id}"

        verbose_logger.debug("Creating IAM role: %s", role_name)

        trust_policy: Final = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": account_id},
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock:{self.aws_region_name}:{account_id}:knowledge-base/*"
                        },
                    },
                }
            ],
        }

        response: Final = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
        )
        role_arn: Final = response["Role"]["Arn"]
        self._created_resources["iam_role"] = role_name

        # Attach permissions policy
        permissions_policy: Final = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": [f"arn:aws:bedrock:{self.aws_region_name}::foundation-model/{self.embedding_model}"],
                },
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [collection_arn],
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{self.s3_bucket}",
                        f"arn:aws:s3:::{self.s3_bucket}/*",
                    ],
                },
            ],
        }

        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}-policy",
            PolicyDocument=json.dumps(permissions_policy),
        )

        # Wait for role to propagate (use asyncio.sleep to avoid blocking)
        await asyncio.sleep(10)

        verbose_logger.info("Created IAM role: %s", role_arn)
        return role_arn

    async def _create_knowledge_base(self, kb_name: str, role_arn: str, collection_arn: str) -> str:
        """Create Bedrock Knowledge Base."""
        bedrock_agent: Final = self._get_boto3_client("bedrock-agent")

        verbose_logger.debug("Creating Knowledge Base: %s", kb_name)

        response: Final = bedrock_agent.create_knowledge_base(
            name=kb_name,
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{self.aws_region_name}::foundation-model/{self.embedding_model}",
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "fieldMapping": {
                        "metadataField": "AMAZON_BEDROCK_METADATA",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "vectorField": "bedrock-knowledge-base-default-vector",
                    },
                    "vectorIndexName": "bedrock-kb-index",
                },
            },
        )
        kb_id: Final = response["knowledgeBase"]["knowledgeBaseId"]
        self._created_resources["knowledge_base"] = kb_id

        # Wait for KB to be active (use asyncio.sleep to avoid blocking)
        verbose_logger.debug("Waiting for Knowledge Base to be active...")
        for _ in range(30):
            kb_status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
            status = kb_status["knowledgeBase"]["status"]
            if status == "ACTIVE":
                break
            await asyncio.sleep(2)
        else:
            raise TimeoutError("Knowledge Base did not become active in time")

        verbose_logger.info("Created Knowledge Base: %s", kb_id)
        return kb_id

    def _create_data_source(self, kb_name: str) -> str:
        """Create Data Source for the Knowledge Base."""
        bedrock_agent: Final = self._get_boto3_client("bedrock-agent")

        verbose_logger.debug("Creating Data Source for KB: %s", self.knowledge_base_id)

        response: Final = bedrock_agent.create_data_source(
            knowledgeBaseId=self.knowledge_base_id,
            name=f"{kb_name}-s3-source",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{self.s3_bucket}",
                    "inclusionPrefixes": [self.s3_prefix],
                },
            },
        )
        ds_id: Final = response["dataSource"]["dataSourceId"]
        self._created_resources["data_source"] = ds_id

        verbose_logger.info("Created Data Source: %s", ds_id)
        return ds_id

    def _get_boto3_client(self, service_name: str):
        """Get a boto3 client for the specified service using BaseAWSLLM auth."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for Bedrock ingestion. Install with: pip install boto3")

        # Get credentials using BaseAWSLLM's get_credentials method
        credentials: Final = self.get_credentials(
            aws_access_key_id=_get_str_or_none(self.vector_store_config.get("aws_access_key_id")),
            aws_secret_access_key=_get_str_or_none(self.vector_store_config.get("aws_secret_access_key")),
            aws_session_token=_get_str_or_none(self.vector_store_config.get("aws_session_token")),
            aws_region_name=self.aws_region_name,
            aws_session_name=_get_str_or_none(self.vector_store_config.get("aws_session_name")),
            aws_profile_name=_get_str_or_none(self.vector_store_config.get("aws_profile_name")),
            aws_role_name=_get_str_or_none(self.vector_store_config.get("aws_role_name")),
            aws_web_identity_token=_get_str_or_none(self.vector_store_config.get("aws_web_identity_token")),
            aws_sts_endpoint=_get_str_or_none(self.vector_store_config.get("aws_sts_endpoint")),
            aws_external_id=_get_str_or_none(self.vector_store_config.get("aws_external_id")),
        )

        # Create session with credentials
        session: Final = boto3.Session(
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=self.aws_region_name,
        )

        return session.client(service_name)

    async def embed(
        self,
        chunks: list[str],
    ) -> list[list[float]] | None:
        """
        Bedrock handles embedding internally - skip this step.

        Returns:
            None (Bedrock embeds when files are ingested)
        """
        return None

    async def store(
        self,
        file_content: bytes | None,
        filename: str | None,
        content_type: str | None,
        chunks: list[str],
        embeddings: list[list[float]] | None,
        existing_file_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Store content in Bedrock Knowledge Base.

        Bedrock workflow:
        1. Auto-detect data source and S3 bucket (if not provided)
        2. Upload file to S3 bucket
        3. Start ingestion job
        4. (Optional) Wait for ingestion to complete

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - Bedrock handles chunking
            embeddings: Ignored - Bedrock handles embedding
            existing_file_id: Existing provider file ID, unsupported for Bedrock

        Returns:
            Tuple of (knowledge_base_id, file_key)
        """
        # Auto-detect data source and S3 bucket if needed
        await self._ensure_config_initialized()

        if not file_content or not filename:
            verbose_logger.warning("No file content or filename provided for Bedrock ingestion")
            return _get_str_or_none(self.knowledge_base_id), None

        # Step 1: Upload file to S3
        s3_client: Final = self._get_boto3_client("s3")
        s3_key: Final = f"{self.s3_prefix.rstrip('/')}/{filename}"

        verbose_logger.debug("Uploading file to s3://%s/%s", self.s3_bucket, s3_key)
        s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type or "application/octet-stream",
        )
        verbose_logger.info("Uploaded file to s3://%s/%s", self.s3_bucket, s3_key)

        # Step 2: Start ingestion job
        bedrock_agent: Final = self._get_boto3_client("bedrock-agent")

        verbose_logger.debug("Starting ingestion job for KB=%s, DS=%s", self.knowledge_base_id, self.data_source_id)
        ingestion_response: Final = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=self.knowledge_base_id,
            dataSourceId=self.data_source_id,
        )
        job_id: Final = ingestion_response["ingestionJob"]["ingestionJobId"]
        verbose_logger.info("Started ingestion job: %s", job_id)

        # Step 3: Wait for ingestion (optional) - use asyncio.sleep to avoid blocking
        if self.wait_for_ingestion:
            import time as time_module

            start_time: Final = time_module.time()
            while time_module.time() - start_time < self.ingestion_timeout:
                job_status = bedrock_agent.get_ingestion_job(
                    knowledgeBaseId=self.knowledge_base_id,
                    dataSourceId=self.data_source_id,
                    ingestionJobId=job_id,
                )
                status = job_status["ingestionJob"]["status"]
                verbose_logger.debug("Ingestion job %s status: %s", job_id, status)

                if status == "COMPLETE":
                    stats = job_status["ingestionJob"].get("statistics", {})
                    verbose_logger.info(
                        "Ingestion complete: %s docs indexed", stats.get("numberOfNewDocumentsIndexed", 0)
                    )
                    break
                elif status == "FAILED":
                    failure_reasons = job_status["ingestionJob"].get("failureReasons", [])
                    verbose_logger.error("Ingestion failed: %s", failure_reasons)
                    break
                elif status in ("STARTING", "IN_PROGRESS"):
                    await asyncio.sleep(2)
                else:
                    verbose_logger.warning("Unknown ingestion status: %s", status)
                    break

        return str(self.knowledge_base_id) if self.knowledge_base_id else None, s3_key
