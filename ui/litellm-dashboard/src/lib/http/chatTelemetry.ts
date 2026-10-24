// Public ingestion token shared with the docs PostHog project.
const POSTHOG_PROJECT_TOKEN = "phc_upsFA5iBuDFKnznEdV9pA5HYW8fwsLMJ8pF2p4xZzzpD";

export async function captureChatRequest(userId: string, requestId: string): Promise<void> {
  const payload = {
    api_key: POSTHOG_PROJECT_TOKEN,
    event: "litellm_chat_request",
    distinct_id: userId,
    uuid: requestId,
    properties: {
      product: "LiteLLM Chat UI",
      $process_person_profile: false,
      $geoip_disable: true,
      $ip: "0.0.0.0",
    },
  };
  const request: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "omit",
    referrerPolicy: "no-referrer",
    keepalive: true,
    signal: AbortSignal.timeout(5000),
    body: JSON.stringify(payload),
  };
  const response = await fetch("https://us.i.posthog.com/i/v0/e/", request);
  if (!response.ok) throw new Error("Chat telemetry was not accepted");
}
