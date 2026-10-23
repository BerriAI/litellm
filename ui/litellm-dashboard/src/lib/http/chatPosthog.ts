// Public ingestion token shared with BerriAI/litellm-docs (posthog-docusaurus).
const PROJECT_TOKEN = "phc_upsFA5iBuDFKnznEdV9pA5HYW8fwsLMJ8pF2p4xZzzpD";

export async function captureChatRequest(distinctId: string, timestamp: string): Promise<void> {
  const request: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "omit",
    referrerPolicy: "no-referrer",
    keepalive: true,
    signal: AbortSignal.timeout(5000),
    body: JSON.stringify({
      api_key: PROJECT_TOKEN,
      batch: [
        {
          event: "LiteLLM Chat UI",
          distinct_id: distinctId,
          timestamp,
          properties: { $process_person_profile: false, $geoip_disable: true },
        },
      ],
    }),
  };
  const response = await fetch("https://us.i.posthog.com/batch/", request);
  if (!response.ok) throw new Error("Chat telemetry was not accepted");
}
