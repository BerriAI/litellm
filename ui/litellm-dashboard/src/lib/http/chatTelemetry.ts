export async function captureChatRequest(userId: string, requestId: string): Promise<void> {
  const request: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "omit",
    referrerPolicy: "no-referrer",
    keepalive: true,
    signal: AbortSignal.timeout(5000),
    body: JSON.stringify({
      user_id: userId,
      request_id: requestId,
    }),
  };
  const response = await fetch("https://chat-telemetry.litellm.ai/v1/chat/events", request);
  if (!response.ok) throw new Error("Chat telemetry was not accepted");
}
