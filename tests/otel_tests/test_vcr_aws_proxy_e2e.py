import io
import openai
import os
import pytest
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
    event_attributes as EventAttributes,
)

from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.proxy.proxy_cli import run_server
from litellm.proxy.proxy_server import ProxyConfig

BEDROCK_MODEL_ARN: str = "arn:aws:bedrock:us-west-2:1234567890123:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
SYSTEM: str = "bedrock"

HERE = os.path.dirname(__file__)
CASSETTE_DIR = os.path.join(HERE, "cassettes")
os.makedirs(CASSETTE_DIR, exist_ok=True)

@pytest.fixture(scope="session")
def span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()

@pytest.fixture(scope="session")
def log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter
    exporter.clear()

@pytest.fixture(scope="session")
def metric_reader():
    exporter = InMemoryMetricReader()
    yield exporter
    # exporter.clear()

@pytest.fixture(scope="session")
def tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider

@pytest.fixture(scope="session")
def event_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return EventLoggerProvider(provider)

@pytest.fixture(scope="session")
def meter_provider(metric_reader):
    return MeterProvider(metric_readers=[metric_reader])

@pytest.fixture(scope="session", autouse=True)
def start_proxy(tracer_provider, event_logger_provider, meter_provider):
    """
    Start the litellm proxy with OTEL instrumentation enabled.
    Configures in-memory OTEL exporters and launches the proxy server.
    """

    ### The instance will add itself to the module litellm.service_callback list to register the OTEL callbacks
    OpenTelemetry(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
        meter_provider=meter_provider,
    )

    config = {
        "model_list": [
            {
                "model_name": "claude-3-7-sonnet",
                "litellm_params": {
                    "model": f"{SYSTEM}/{BEDROCK_MODEL_ARN}",
                    "provider": "aws",
                    "region": "us-west-2"
                },
            }
        ],
        "litellm_settings": {"callbacks": ["otel"]},
    }

    async def get_config_override(self, config_file_path=None):
        return config
    ProxyConfig.get_config = get_config_override
    ProxyConfig._get_config_from_file = lambda self, config_file_path=None: config
    config_path = 'dummy'

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exceptions = []

    def target():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                run_server.main(
                    args=["-c", config_path],
                    standalone_mode=False
                )
        except Exception as e:
            exceptions.append((e, traceback.format_exc()))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    time.sleep(5)

    print("Proxy alive:", thread.is_alive())
    print("Proxy stdout:\n", stdout_buf.getvalue())
    print("Proxy stderr:\n", stderr_buf.getvalue())
    if exceptions:
        err, tb = exceptions[0]
        pytest.fail(f"Proxy startup failed with {err}\n{tb}")

# ─── HELPERS ────────────────────────────────────────────────────────────────

def call_proxy_and_get_response():
    client = openai.AsyncOpenAI(
        api_key="unused-for-aws-bedrock",
        base_url="http://0.0.0.0:4000",
    )
    return client.chat.completions.create(
        model="claude-3-7-sonnet",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=False,
    )


def assert_response_valid(response):
    assert hasattr(response, "choices")
    assert response.choices and response.choices[0].message.content


def get_genai_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if any(k.startswith("gen_ai.") for k in s.attributes)]


def assert_spans_have_expected(spans):
    assert spans, "Expected at least one gen_ai span"
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == SYSTEM
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == BEDROCK_MODEL_ARN
    assert span.attributes["gen_ai.completion.0.finish_reason"] == "stop"


def get_event_logs(log_exporter, event_name):
    return [
        log for log in log_exporter.get_finished_logs()
        if log.log_record.attributes[EventAttributes.EVENT_NAME] == event_name
    ]


def assert_logs_correct(log_exporter, response):
    user_logs = get_event_logs(log_exporter, "gen_ai.user.message")
    choice_logs = get_event_logs(log_exporter, "gen_ai.choice")

    assert user_logs, "User message log not found"
    assert user_logs[0].log_record.attributes[GenAIAttributes.GEN_AI_SYSTEM] == SYSTEM
    assert user_logs[0].log_record.body["content"] == "What is the capital of France?"

    assert choice_logs, "Choice log not found"
    cl = choice_logs[0]
    assert cl.log_record.attributes[GenAIAttributes.GEN_AI_SYSTEM] == SYSTEM
    assert cl.log_record.body["message"]["content"] == response.choices[0].message.content
    assert cl.log_record.body["finish_reason"] == "stop"


def find_metric(metrics_data, name):
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    return m
    return None


def wait_for_metric(metric_reader, name, timeout=10.0, interval=0.1):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = metric_reader.get_metrics_data()
        if find_metric(last, name):
            return last, find_metric(last, name)
        time.sleep(interval)
    return last, None


def assert_metric_has_attr(metric, attr, expected):
    assert any(dp.attributes.get(attr) == expected for dp in metric.data.data_points), (
        f"{metric.name!r} has no data-point with {attr}=={expected!r}"
    )

# ─── TESTS ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.vcr(
    cassette_library_dir=CASSETTE_DIR,
    record_mode="once",
    ignore_localhost=True,
)
async def test_litellm_proxy_otel_telemetry(span_exporter, log_exporter, metric_reader):
    # 1) call the proxy and await the response
    response = await call_proxy_and_get_response()
    assert_response_valid(response)

    # 2) spans
    spans = get_genai_spans(span_exporter)
    assert_spans_have_expected(spans)

    # 3) logs
    assert_logs_correct(log_exporter, response)

    # 4) metrics (wait up to 10s)
    _, op_metric = wait_for_metric(metric_reader, "gen_ai.client.operation.duration")
    assert op_metric, "Request duration metric not found"
    assert_metric_has_attr(op_metric, GenAIAttributes.GEN_AI_REQUEST_MODEL, BEDROCK_MODEL_ARN)

    # 5) optionally check token usage
    _, tok_metric = wait_for_metric(metric_reader, "gen_ai.client.token.usage")
    if tok_metric:
        assert_metric_has_attr(tok_metric, GenAIAttributes.GEN_AI_REQUEST_MODEL, BEDROCK_MODEL_ARN)