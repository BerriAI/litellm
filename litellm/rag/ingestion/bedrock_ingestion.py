"""
Bedrock-specific RAG Ingestion implementation.

Bedrock Knowledge Bases handle embedding internally when files are ingested,
so this implementation uploads files to S3 and triggers ingestion jobs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class BedrockRAGIngestion(BaseRAGIngestion, BaseAWSLLM):
    """
    Bedrock Knowledge Base RAG ingestion.

    Key differences from base:
    - Embedding is handled by Bedrock when files are ingested
    - Files are uploaded to S3 and ingestion jobs are triggered
    - Requires existing Knowledge Base and Data Source to be set up

    Required vector_store config:
    - knowledge_base_id: Bedrock KB ID
    - data_source_id: Data source ID
    - s3_bucket: S3 bucket name for file uploads
    - s3_prefix: (optional) S3 key prefix, defaults to "data/"

    Optional config (uses BaseAWSLLM auth):
    - aws_access_key_id: AWS access key
    - aws_secret_access_key: AWS secret key
    - aws_session_token: AWS session token
    - aws_region_name: AWS region (defaults to us-west-2)
    - aws_role_name: IAM role to assume
    - aws_session_name: Session name for role assumption
    - aws_profile_name: AWS profile name
    - aws_web_identity_token: Web identity token for IRSA
    - aws_sts_endpoint: Custom STS endpoint
    - aws_external_id: External ID for role assumption
    - wait_for_ingestion: Whether to wait for ingestion to complete (default: True)
    - ingestion_timeout: Max seconds to wait for ingestion (default: 300)
    """

    def __init__(
        self,
        ingest_options: "RAGIngestOptions",
        router: Optional["Router"] = None,
    ):
        BaseRAGIngestion.__init__(self, ingest_options=ingest_options, router=router)
        BaseAWSLLM.__init__(self)

        # Bedrock-specific config
        self.knowledge_base_id = self.vector_store_config.get("knowledge_base_id")
        self.data_source_id = self.vector_store_config.get("data_source_id")
        self.s3_bucket = self.vector_store_config.get("s3_bucket")
        self.s3_prefix = self.vector_store_config.get("s3_prefix", "data/")
        self.wait_for_ingestion = self.vector_store_config.get("wait_for_ingestion", True)
        self.ingestion_timeout = self.vector_store_config.get("ingestion_timeout", 300)

        # Get AWS region using BaseAWSLLM method
        self.aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=self.vector_store_config.get("aws_region_name")
        )

        # Validate required config
        if not self.knowledge_base_id:
            raise ValueError("knowledge_base_id is required for Bedrock ingestion")
        if not self.data_source_id:
            raise ValueError("data_source_id is required for Bedrock ingestion")
        if not self.s3_bucket:
            raise ValueError("s3_bucket is required for Bedrock ingestion")

    def _get_boto3_client(self, service_name: str):
        """Get a boto3 client for the specified service using BaseAWSLLM auth."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for Bedrock ingestion. Install with: pip install boto3")

        # Get credentials using BaseAWSLLM's get_credentials method
        credentials = self.get_credentials(
            aws_access_key_id=self.vector_store_config.get("aws_access_key_id"),
            aws_secret_access_key=self.vector_store_config.get("aws_secret_access_key"),
            aws_session_token=self.vector_store_config.get("aws_session_token"),
            aws_region_name=self.aws_region_name,
            aws_session_name=self.vector_store_config.get("aws_session_name"),
            aws_profile_name=self.vector_store_config.get("aws_profile_name"),
            aws_role_name=self.vector_store_config.get("aws_role_name"),
            aws_web_identity_token=self.vector_store_config.get("aws_web_identity_token"),
            aws_sts_endpoint=self.vector_store_config.get("aws_sts_endpoint"),
            aws_external_id=self.vector_store_config.get("aws_external_id"),
        )

        # Create session with credentials
        session = boto3.Session(
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=self.aws_region_name,
        )

        return session.client(service_name)

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Bedrock handles embedding internally - skip this step.

        Returns:
            None (Bedrock embeds when files are ingested)
        """
        return None

    async def store(
        self,
        file_content: Optional[bytes],
        filename: Optional[str],
        content_type: Optional[str],
        chunks: List[str],
        embeddings: Optional[List[List[float]]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Store content in Bedrock Knowledge Base.

        Bedrock workflow:
        1. Upload file to S3 bucket
        2. Start ingestion job
        3. (Optional) Wait for ingestion to complete

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - Bedrock handles chunking
            embeddings: Ignored - Bedrock handles embedding

        Returns:
            Tuple of (knowledge_base_id, file_key)
        """
        if not file_content or not filename:
            verbose_logger.warning("No file content or filename provided for Bedrock ingestion")
            return self.knowledge_base_id, None

        # Step 1: Upload file to S3
        s3_client = self._get_boto3_client("s3")
        s3_key = f"{self.s3_prefix.rstrip('/')}/{filename}"

        verbose_logger.debug(f"Uploading file to s3://{self.s3_bucket}/{s3_key}")
        s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type or "application/octet-stream",
        )
        verbose_logger.info(f"Uploaded file to s3://{self.s3_bucket}/{s3_key}")

        # Step 2: Start ingestion job
        bedrock_agent = self._get_boto3_client("bedrock-agent")

        verbose_logger.debug(
            f"Starting ingestion job for KB={self.knowledge_base_id}, DS={self.data_source_id}"
        )
        ingestion_response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=self.knowledge_base_id,
            dataSourceId=self.data_source_id,
        )
        job_id = ingestion_response["ingestionJob"]["ingestionJobId"]
        verbose_logger.info(f"Started ingestion job: {job_id}")

        # Step 3: Wait for ingestion (optional)
        if self.wait_for_ingestion:
            start_time = time.time()
            while time.time() - start_time < self.ingestion_timeout:
                job_status = bedrock_agent.get_ingestion_job(
                    knowledgeBaseId=self.knowledge_base_id,
                    dataSourceId=self.data_source_id,
                    ingestionJobId=job_id,
                )
                status = job_status["ingestionJob"]["status"]
                verbose_logger.debug(f"Ingestion job {job_id} status: {status}")

                if status == "COMPLETE":
                    stats = job_status["ingestionJob"].get("statistics", {})
                    verbose_logger.info(
                        f"Ingestion complete: {stats.get('numberOfNewDocumentsIndexed', 0)} docs indexed"
                    )
                    break
                elif status == "FAILED":
                    failure_reasons = job_status["ingestionJob"].get("failureReasons", [])
                    verbose_logger.error(f"Ingestion failed: {failure_reasons}")
                    break
                elif status in ("STARTING", "IN_PROGRESS"):
                    time.sleep(2)
                else:
                    verbose_logger.warning(f"Unknown ingestion status: {status}")
                    break

        return self.knowledge_base_id, s3_key

