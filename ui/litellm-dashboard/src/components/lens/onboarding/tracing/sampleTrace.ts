const NS_PER_MS = BigInt(1_000_000);

export const SAMPLE_TRACE_SERVICE = "litellm-sample-agent";

const randomHexId = (bytes: number): string =>
  Array.from(crypto.getRandomValues(new Uint8Array(bytes)), (b) => b.toString(16).padStart(2, "0")).join("");

const str = (key: string, value: string) => ({ key, value: { stringValue: value } });

const messages = (role: string, content: string): string => JSON.stringify([{ role, content }]);

interface SampleSpan {
  name: string;
  parent: string | null;
  startMs: number;
  endMs: number;
  attributes: ReturnType<typeof str>[];
}

export function sampleTraceExport(nowMs: number): { traceId: string; body: object } {
  const traceId = randomHexId(16);
  const agentId = randomHexId(8);
  const question = "What is the weather in San Francisco?";
  const answer = "It is 18°C and sunny in San Francisco.";
  const spans: readonly (SampleSpan & { id: string })[] = [
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
      id: randomHexId(8),
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
      id: randomHexId(8),
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
    traceId,
    body: {
      resourceSpans: [
        {
          resource: { attributes: [str("service.name", SAMPLE_TRACE_SERVICE)] },
          scopeSpans: [
            {
              scope: { name: "litellm-ui-sample" },
              spans: spans.map((s) => ({
                traceId,
                spanId: s.id,
                ...(s.parent ? { parentSpanId: s.parent } : {}),
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
