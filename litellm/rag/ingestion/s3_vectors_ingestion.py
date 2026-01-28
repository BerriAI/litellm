"""
S3 Vectors-specific RAG Ingestion implementation.

S3 Vectors is AWS's native vector storage service that provides:
- Purpose-built vector buckets for storing and querying vectors
- Vector indexes with configurable dimensions and distance metrics
- Metadata filtering for semantic search

This implementation:
1. Auto-creates vector buckets and indexes if not provided
2. Uses LiteLLM's embedding API (supports any provider)
3. Uses httpx + AWS SigV4 signing (no boto3 dependency for S3 Vectors APIs)
4. Stores vectors with metadata using PutVectors API
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    S3_VECTORS_DEFAULT_DIMENSION,
    S3_VECTORS_DEFAULT_DISTANCE_METRIC,
    S3_VECTORS_DEFAULT_NON_FILTERABLE_METADATA_KEYS,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class S3VectorsRAGIngestion(BaseRAGIngestion, BaseAWSLLM):
    """
    S3 Vectors RAG ingestion using httpx + AWS SigV4 signing.

    Workflow:
    1. Auto-create vector bucket if needed (CreateVectorBucket API)
    2. Auto-create vector index if needed (CreateVectorIndex API)
    3. Generate embeddings using LiteLLM (supports any provider)
    4. Store vectors with PutVectors API

    Configuration:
    - vector_bucket_name: S3 vector bucket name (required)
    - index_name: Vector index name (auto-creates if not provided)
    - dimension: Vector dimension (default: S3_VECTORS_DEFAULT_DIMENSION)
    - distance_metric: "cosine" or "euclidean" (default: S3_VECTORS_DEFAULT_DISTANCE_METRIC)
    - non_filterable_metadata_keys: List of metadata keys to exclude from filtering
    """

    def __init__(
        self,
        ingest_options: "RAGIngestOptions",
        router: Optional["Router"] = None,
    ):
        BaseRAGIngestion.__init__(self, ingest_options=ingest_options, router=router)
        BaseAWSLLM.__init__(self)

        # Extract config
        self.vector_bucket_name = self.vector_store_config["vector_bucket_name"]
        self.index_name = self.vector_store_config.get("index_name")
        self.distance_metric = self.vector_store_config.get(
            "distance_metric", S3_VECTORS_DEFAULT_DISTANCE_METRIC
        )
        self.non_filterable_metadata_keys = self.vector_store_config.get(
            "non_filterable_metadata_keys",
            S3_VECTORS_DEFAULT_NON_FILTERABLE_METADATA_KEYS,
        )
        
        # Get dimension from config (will be auto-detected on first use if not provided)
        self.dimension = self._get_dimension_from_config()

        # Get AWS region using BaseAWSLLM method
        _aws_region = self.vector_store_config.get("aws_region_name")
        self.aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=str(_aws_region) if _aws_region else None
        )

        # Create httpx client (similar to s3_v2.py)
        ssl_verify = self._get_ssl_verify(
            ssl_verify=self.vector_store_config.get("ssl_verify")
        )
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"ssl_verify": ssl_verify} if ssl_verify is not None else None,
        )

        # Track if infrastructure is initialized
        self._config_initialized = False

    async def _get_dimension_from_embedding_request(self) -> int:
        """
        Auto-detect dimension by making a test embedding request.
        
        Makes a single embedding request with a test string to determine
        the output dimension of the embedding model.
        """
        if not self.embedding_config or "model" not in self.embedding_config:
            return S3_VECTORS_DEFAULT_DIMENSION
        
        try:
            model_name = self.embedding_config["model"]
            verbose_logger.debug(
                f"Auto-detecting dimension by making test embedding request to {model_name}"
            )
            
            # Make a test embedding request
            test_input = "test"
            if self.router:
                response = await self.router.aembedding(model=model_name, input=[test_input])
            else:
                response = await litellm.aembedding(model=model_name, input=[test_input])
            
            # Get dimension from the response
            if response.data and len(response.data) > 0:
                dimension = len(response.data[0]["embedding"])
                verbose_logger.debug(
                    f"Auto-detected dimension {dimension} for embedding model {model_name}"
                )
                return dimension
        except Exception as e:
            verbose_logger.warning(
                f"Could not auto-detect dimension from embedding model: {e}. "
                f"Using default dimension of {S3_VECTORS_DEFAULT_DIMENSION}."
            )
        
        return S3_VECTORS_DEFAULT_DIMENSION
    
    def _get_dimension_from_config(self) -> Optional[int]:
        """
        Get vector dimension from config if explicitly provided.
        
        Returns None if dimension should be auto-detected.
        """
        if "dimension" in self.vector_store_config:
            return int(self.vector_store_config["dimension"])
        return None

    async def _ensure_config_initialized(self):
        """Lazily initialize S3 Vectors infrastructure."""
        if self._config_initialized:
            return

        # Auto-detect dimension if not provided
        if self.dimension is None:
            self.dimension = await self._get_dimension_from_embedding_request()

        # Ensure vector bucket exists
        await self._ensure_vector_bucket_exists()

        # Ensure vector index exists
        if not self.index_name:
            # Auto-generate index name
            unique_id = uuid.uuid4().hex[:8]
            self.index_name = f"litellm-index-{unique_id}"

        await self._ensure_vector_index_exists()

        self._config_initialized = True

    async def _sign_and_execute_request(
        self,
        method: str,
        url: str,
        data: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Helper to sign and execute AWS API requests using httpx + SigV4.

        Pattern from litellm/integrations/s3_v2.py
        """
        try:
            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError(
                "Missing botocore to call S3 Vectors. Run 'pip install boto3'."
            )

        # Get AWS credentials using BaseAWSLLM's get_credentials method
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

        # Prepare headers
        if headers is None:
            headers = {}

        if data:
            headers["Content-Type"] = "application/json"
            # Calculate SHA256 hash of the content
            content_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
            headers["x-amz-content-sha256"] = content_hash
        else:
            # For requests without body, use hash of empty string
            headers["x-amz-content-sha256"] = hashlib.sha256(b"").hexdigest()

        # Prepare the request
        req = requests.Request(method, url, data=data, headers=headers)
        prepped = req.prepare()

        # Sign the request
        aws_request = AWSRequest(
            method=prepped.method,
            url=prepped.url,
            data=prepped.body,
            headers=prepped.headers,
        )
        SigV4Auth(credentials, "s3vectors", self.aws_region_name).add_auth(aws_request)

        # Prepare the signed headers
        signed_headers = dict(aws_request.headers.items())

        # Make the request using specific method (pattern from s3_v2.py)
        method_upper = method.upper()
        if method_upper == "PUT":
            response = await self.async_httpx_client.put(
                url, data=data, headers=signed_headers
            )
        elif method_upper == "POST":
            response = await self.async_httpx_client.post(
                url, data=data, headers=signed_headers
            )
        elif method_upper == "GET":
            response = await self.async_httpx_client.get(url, headers=signed_headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    async def _ensure_vector_bucket_exists(self):
        """Create vector bucket if it doesn't exist using GetVectorBucket and CreateVectorBucket APIs."""
        verbose_logger.debug(
            f"Ensuring S3 vector bucket exists: {self.vector_bucket_name}"
        )
        
        # Validate bucket name (AWS S3 naming rules)
        if len(self.vector_bucket_name) < 3:
            raise ValueError(
                f"Invalid vector_bucket_name '{self.vector_bucket_name}': "
                f"AWS S3 bucket names must be at least 3 characters long. "
                f"Please provide a valid bucket name (e.g., 'my-vector-bucket')."
            )
        if not self.vector_bucket_name.replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                f"Invalid vector_bucket_name '{self.vector_bucket_name}': "
                f"AWS S3 bucket names can only contain lowercase letters, numbers, hyphens, and periods. "
                f"Please provide a valid bucket name (e.g., 'my-vector-bucket')."
            )

        # Try to get bucket info using GetVectorBucket API
        get_url = f"https://s3vectors.{self.aws_region_name}.api.aws/GetVectorBucket"
        get_body = safe_dumps({"vectorBucketName": self.vector_bucket_name})

        try:
            response = await self._sign_and_execute_request("POST", get_url, data=get_body)
            if response.status_code == 200:
                verbose_logger.debug(f"Vector bucket {self.vector_bucket_name} exists")
                return
        except Exception as e:
            verbose_logger.debug(
                f"Bucket check failed (may not exist): {e}, attempting to create"
            )

        # Create vector bucket using CreateVectorBucket API
        try:
            verbose_logger.debug(f"Creating vector bucket: {self.vector_bucket_name}")
            create_url = f"https://s3vectors.{self.aws_region_name}.api.aws/CreateVectorBucket"
            create_body = safe_dumps({
                "vectorBucketName": self.vector_bucket_name
            })
            
            response = await self._sign_and_execute_request("POST", create_url, data=create_body)

            if response.status_code in (200, 201):
                verbose_logger.info(f"Created vector bucket: {self.vector_bucket_name}")
            elif response.status_code == 409:
                # Bucket already exists (ConflictException)
                verbose_logger.debug(
                    f"Vector bucket {self.vector_bucket_name} already exists"
                )
            else:
                verbose_logger.error(f"CreateVectorBucket failed: {response.status_code} - {response.text}")
                response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error creating vector bucket: {e}")
            raise

    async def _ensure_vector_index_exists(self):
        """Create vector index if it doesn't exist using GetIndex and CreateIndex APIs."""
        verbose_logger.debug(
            f"Ensuring vector index exists: {self.vector_bucket_name}/{self.index_name}"
        )

        # Try to get index info using GetIndex API
        get_url = f"https://s3vectors.{self.aws_region_name}.api.aws/GetIndex"
        get_body = safe_dumps({
            "vectorBucketName": self.vector_bucket_name,
            "indexName": self.index_name
        })

        try:
            response = await self._sign_and_execute_request("POST", get_url, data=get_body)
            if response.status_code == 200:
                verbose_logger.debug(f"Vector index {self.index_name} exists")
                return
        except Exception as e:
            verbose_logger.debug(
                f"Index check failed (may not exist): {e}, attempting to create"
            )

        # Create vector index using CreateIndex API
        try:
            verbose_logger.debug(
                f"Creating vector index: {self.index_name} with dimension={self.dimension}, metric={self.distance_metric}"
            )

            # Prepare index configuration per AWS API docs
            index_config = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "dataType": "float32",
                "dimension": self.dimension,
                "distanceMetric": self.distance_metric,
            }

            if self.non_filterable_metadata_keys:
                index_config["metadataConfiguration"] = {
                    "nonFilterableMetadataKeys": self.non_filterable_metadata_keys
                }

            create_url = f"https://s3vectors.{self.aws_region_name}.api.aws/CreateIndex"
            response = await self._sign_and_execute_request(
                "POST", create_url, data=safe_dumps(index_config)
            )

            if response.status_code in (200, 201):
                verbose_logger.info(f"Created vector index: {self.index_name}")
            elif response.status_code == 409:
                verbose_logger.debug(f"Vector index {self.index_name} already exists")
            else:
                verbose_logger.error(f"CreateIndex failed: {response.status_code} - {response.text}")
                response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error creating vector index: {e}")
            raise

    async def _put_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Call PutVectors API to store vectors in S3 Vectors.

        Args:
            vectors: List of vector objects with keys: "key", "data", "metadata"
        """
        verbose_logger.debug(
            f"Storing {len(vectors)} vectors in {self.vector_bucket_name}/{self.index_name}"
        )

        url = f"https://s3vectors.{self.aws_region_name}.api.aws/PutVectors"

        # Prepare request body per AWS API docs
        request_body = {
            "vectorBucketName": self.vector_bucket_name,
            "indexName": self.index_name,
            "vectors": vectors
        }

        try:
            response = await self._sign_and_execute_request(
                "POST", url, data=safe_dumps(request_body)
            )

            if response.status_code in (200, 201):
                verbose_logger.info(
                    f"Successfully stored {len(vectors)} vectors in index {self.index_name}"
                )
            else:
                verbose_logger.error(
                    f"PutVectors failed with status {response.status_code}: {response.text}"
                )
                response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error storing vectors: {e}")
            raise

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Generate embeddings using LiteLLM's embedding API.

        Supports any embedding provider (OpenAI, Bedrock, Cohere, etc.)
        """
        if not chunks:
            return None

        # Use embedding config from ingest_options or default
        if not self.embedding_config:
            verbose_logger.warning(
                "No embedding config provided, using default text-embedding-3-small"
            )
            self.embedding_config = {"model": "text-embedding-3-small"}

        embedding_model = self.embedding_config.get("model", "text-embedding-3-small")

        verbose_logger.debug(
            f"Generating embeddings for {len(chunks)} chunks using {embedding_model}"
        )

        # Convert to list to ensure type compatibility
        input_chunks: List[str] = list(chunks)
        
        if self.router:
            response = await self.router.aembedding(model=embedding_model, input=input_chunks)
        else:
            response = await litellm.aembedding(model=embedding_model, input=input_chunks)

        return [item["embedding"] for item in response.data]

    async def store(
        self,
        file_content: Optional[bytes],
        filename: Optional[str],
        content_type: Optional[str],
        chunks: List[str],
        embeddings: Optional[List[List[float]]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Store vectors in S3 Vectors using PutVectors API.

        Steps:
        1. Ensure vector bucket exists (auto-create if needed)
        2. Ensure vector index exists (auto-create if needed)
        3. Prepare vector data with metadata
        4. Call PutVectors API with httpx + SigV4 signing

        Args:
            file_content: Raw file bytes (not used for S3 Vectors)
            filename: Name of the file
            content_type: MIME type (not used for S3 Vectors)
            chunks: Text chunks
            embeddings: Vector embeddings

        Returns:
            Tuple of (index_name, filename)
        """
        # Ensure infrastructure exists
        await self._ensure_config_initialized()

        if not embeddings or not chunks:
            error_msg = (
                "No text content could be extracted from the file for embedding. "
                "Possible causes:\n"
                "  1. PDF files require OCR - add 'ocr' config with a vision model (e.g., 'anthropic/claude-3-5-sonnet-20241022')\n"
                "  2. Binary files cannot be processed - convert to text first\n"
                "  3. File is empty or contains no extractable text\n"
                "For PDFs, either enable OCR or use a PDF extraction library to convert to text before ingestion."
            )
            verbose_logger.error(error_msg)
            raise ValueError(error_msg)

        # Prepare vectors for PutVectors API
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Build metadata dict
            metadata: Dict[str, str] = {
                "source_text": chunk,  # Non-filterable (for reference)
                "chunk_index": str(i),  # Filterable
            }
            
            if filename:
                metadata["filename"] = filename  # Filterable
            
            vector_obj = {
                "key": f"{filename}_{i}" if filename else f"chunk_{i}",
                "data": {"float32": embedding},
                "metadata": metadata,
            }

            vectors.append(vector_obj)

        # Call PutVectors API
        await self._put_vectors(vectors)

        # Return vector_store_id in format bucket_name:index_name for S3 Vectors search compatibility
        vector_store_id = f"{self.vector_bucket_name}:{self.index_name}"
        return vector_store_id, filename

    async def query_vector_store(
        self, vector_store_id: str, query: str, top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Query S3 Vectors using QueryVectors API.

        Args:
            vector_store_id: Index name
            query: Query text
            top_k: Number of results to return

        Returns:
            Query results with vectors and metadata
        """
        verbose_logger.debug(f"Querying index {vector_store_id} with query: {query}")

        # Generate query embedding
        if not self.embedding_config:
            self.embedding_config = {"model": "text-embedding-3-small"}

        embedding_model = self.embedding_config.get("model", "text-embedding-3-small")

        response = await litellm.aembedding(model=embedding_model, input=[query])
        query_embedding = response.data[0]["embedding"]

        # Call QueryVectors API
        url = f"https://s3vectors.{self.aws_region_name}.api.aws/QueryVectors"

        request_body = {
            "vectorBucketName": self.vector_bucket_name,
            "indexName": vector_store_id,
            "queryVector": {"float32": query_embedding},
            "topK": top_k,
            "returnDistance": True,
            "returnMetadata": True,
        }

        try:
            response = await self._sign_and_execute_request(
                "POST", url, data=safe_dumps(request_body)
            )

            if response.status_code == 200:
                results = response.json()
                verbose_logger.debug(f"Query returned {len(results.get('vectors', []))} results")

                # Check if query terms appear in results
                if results.get("vectors"):
                    for result in results["vectors"]:
                        metadata = result.get("metadata", {})
                        source_text = metadata.get("source_text", "")
                        if query.lower() in source_text.lower():
                            return results

                # Return results even if exact match not found
                return results
            else:
                verbose_logger.error(
                    f"QueryVectors failed with status {response.status_code}: {response.text}"
                )
                return None
        except Exception as e:
            verbose_logger.exception(f"Error querying vectors: {e}")
            return None
