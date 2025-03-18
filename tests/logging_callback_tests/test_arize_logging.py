import os
import sys
import time
from unittest.mock import Mock, patch

from litellm.main import completion
import opentelemetry.exporter.otlp.proto.grpc.trace_exporter

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import litellm


def test_arize_callback():
    litellm.callbacks = ["arize"]
    os.environ["ARIZE_SPACE_KEY"] = "test_space_key"
    os.environ["ARIZE_API_KEY"] = "test_api_key"
    os.environ["ARIZE_ENDPOINT"] = "https://otlp.arize.com/v1"

    os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "1"  
    os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = "1"   
    os.environ["OTEL_BSP_SCHEDULE_DELAY_MILLIS"] = "1" 
    os.environ["OTEL_BSP_EXPORT_TIMEOUT_MILLIS"] = "5" 

    with patch.object(
        opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter,
        'export',
        new=Mock()
    ) as patched_export:        
        completion(
            model="openai/test-model",
            messages=[{"role": "user", "content": "arize test content"}],
            stream=False,
            mock_response="hello there!",
        )

        time.sleep(1)
        assert patched_export.called
