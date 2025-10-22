from openai.types.batch import BatchRequestCounts
from openai.types.batch import Metadata as OpenAIBatchMetadata

from litellm.types.utils import LiteLLMBatch


class BedrockBatchesHandler:
    """
    Handler for Bedrock Batches.

    Specific providers/models needed some special handling.

    E.g. Twelve Labs Embedding Async Invoke
    """
    @staticmethod
    def _handle_async_invoke_status(
        batch_id: str, aws_region_name: str, logging_obj=None, **kwargs
    ) -> "LiteLLMBatch":
        """
        Handle async invoke status check for AWS Bedrock.

        This is for Twelve Labs Embedding Async Invoke.

        Args:
            batch_id: The async invoke ARN
            aws_region_name: AWS region name
            **kwargs: Additional parameters

        Returns:
            dict: Status information including status, output_file_id (S3 URL), etc.
        """
        import asyncio

        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding

        async def _async_get_status():
            # Create embedding handler instance
            embedding_handler = BedrockEmbedding()

            # Get the status of the async invoke job
            status_response = await embedding_handler._get_async_invoke_status(
                invocation_arn=batch_id,
                aws_region_name=aws_region_name,
                logging_obj=logging_obj,
                **kwargs,
            )

            # Transform response to a LiteLLMBatch object
            from litellm.types.utils import LiteLLMBatch

            openai_batch_metadata: OpenAIBatchMetadata = {
                "output_file_id": status_response["outputDataConfig"][
                    "s3OutputDataConfig"
                ]["s3Uri"],
                "failure_message": status_response.get("failureMessage") or "",
                "model_arn": status_response["modelArn"],
            }

            result = LiteLLMBatch(
                id=status_response["invocationArn"],
                object="batch",
                status=status_response["status"],
                created_at=status_response["submitTime"],
                in_progress_at=status_response["lastModifiedTime"],
                completed_at=status_response.get("endTime"),
                failed_at=status_response.get("endTime")
                if status_response["status"] == "failed"
                else None,
                request_counts=BatchRequestCounts(
                    total=1,
                    completed=1 if status_response["status"] == "completed" else 0,
                    failed=1 if status_response["status"] == "failed" else 0,
                ),
                metadata=openai_batch_metadata,
                completion_window="24h",
                endpoint="/v1/embeddings",
                input_file_id="",
            )

            return result

        # Since this function is called from within an async context via run_in_executor,
        # we need to create a new event loop in a thread to avoid conflicts
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_async_get_status())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
