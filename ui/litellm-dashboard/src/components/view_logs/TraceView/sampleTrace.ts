const NS_PER_MS = BigInt(1_000_000);

export const SAMPLE_TRACE_SERVICE = "litellm-sample-agent";

interface OtlpId {
  hex: string;
  base64: string;
}

// TODO: the proxy's OTLP/JSON decoder base64-decodes ids; the OTLP spec says hex. Switch to hex once that's fixed.
const randomId = (bytes: number): OtlpId => {
  const raw = crypto.getRandomValues(new Uint8Array(bytes));
  return {
    hex: Array.from(raw, (b) => b.toString(16).padStart(2, "0")).join(""),
    base64: btoa(String.fromCharCode(...raw)),
  };
};

const str = (key: string, value: string) => ({ key, value: { stringValue: value } });

const messages = (role: string, content: string): string => JSON.stringify([{ role, content }]);

interface SampleSpan {
  name: string;
  parent: OtlpId | null;
  startMs: number;
  endMs: number;
  attributes: ReturnType<typeof str>[];
}

/** One small agent run (agent -> LLM call -> tool call) as an OTLP/JSON export request. */
export function sampleTraceExport(nowMs: number): { traceId: string; body: object } {
  const traceId = randomId(16);
  const agentId = randomId(8);
  const question = "What is the weather in San Francisco?";
  const answer = "It is 18°C and sunny in San Francisco.";
  const spans: readonly (SampleSpan & { id: OtlpId })[] = [
    {
      id: agentId,
      name: "weather_agent",
      parent: null,
      startMs: 0,
      endMs: 2400,
      attributes: [
        str("gen_ai.operation.name", "invoke_agent"),
        str("gen_ai.agent.name", "weather_agent"),
        str("gen_ai.input.messages", messages("user", question)),
        str("gen_ai.output.messages", messages("assistant", answer)),
      ],
    },
    {
      id: randomId(8),
      name: "chat sample-model",
      parent: agentId,
      startMs: 100,
      endMs: 1500,
      attributes: [
        str("gen_ai.operation.name", "chat"),
        str("gen_ai.agent.name", "weather_agent"),
        str("gen_ai.request.model", "sample-model"),
        str("gen_ai.input.messages", messages("user", question)),
        str("gen_ai.output.messages", messages("assistant", "Calling get_weather(city=San Francisco)")),
      ],
    },
    {
      id: randomId(8),
      name: "get_weather",
      parent: agentId,
      startMs: 1600,
      endMs: 2300,
      attributes: [
        str("gen_ai.operation.name", "execute_tool"),
        str("gen_ai.agent.name", "weather_agent"),
        str("gen_ai.tool.call.arguments", JSON.stringify({ city: "San Francisco" })),
        str("gen_ai.tool.call.result", JSON.stringify({ temp_c: 18, sky: "sunny" })),
      ],
    },
  ];
  const startNs = BigInt(nowMs - 2400) * NS_PER_MS;
  const toNs = (ms: number): string => String(startNs + BigInt(ms) * NS_PER_MS);
  return {
    traceId: traceId.hex,
    body: {
      resourceSpans: [
        {
          resource: { attributes: [str("service.name", SAMPLE_TRACE_SERVICE)] },
          scopeSpans: [
            {
              scope: { name: "litellm-ui-sample" },
              spans: spans.map((s) => ({
                traceId: traceId.base64,
                spanId: s.id.base64,
                ...(s.parent ? { parentSpanId: s.parent.base64 } : {}),
                name: s.name,
                kind: 1,
                startTimeUnixNano: toNs(s.startMs),
                endTimeUnixNano: toNs(s.endMs),
                attributes: s.attributes,
                status: { code: 1 },
              })),
            },
          ],
        },
      ],
    },
  };
}
