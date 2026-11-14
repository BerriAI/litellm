import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import {
  ERROR_SUMMARY_LIMIT,
  createGatewayClient,
  gatewayConfigFrom,
  gatewayRoot,
  modelGroupInfoUrl,
  openAiBaseUrl,
  summarizeErrorBody,
  USER_AGENT,
  type GatewayConfig,
} from "../src/gateway";
import { buildChatCompletionParams } from "../src/messages";

interface RecordedRequest {
  readonly method: string | undefined;
  readonly url: string | undefined;
  readonly authorization: string | undefined;
  readonly userAgent: string | undefined;
  readonly body: string;
}

type Handler = (request: RecordedRequest, response: ServerResponse) => void;

const readBody = (request: IncomingMessage): Promise<string> =>
  new Promise((resolve) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });

const servers: Server[] = [];

const startGateway = (handler: Handler): Promise<{ readonly url: string; readonly requests: readonly RecordedRequest[] }> =>
  new Promise((resolve) => {
    const requests: RecordedRequest[] = [];
    const server = createServer(async (request, response) => {
      const recorded: RecordedRequest = {
        method: request.method,
        url: request.url,
        authorization: request.headers.authorization,
        userAgent: request.headers["user-agent"],
        body: await readBody(request),
      };
      requests.push(recorded);
      handler(recorded, response);
    });
    servers.push(server);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({ url: `http://127.0.0.1:${port}`, requests });
    });
  });

afterEach(() => {
  servers.splice(0).forEach((server) => server.close());
});

const configFor = (baseUrl: string, apiKey: string): GatewayConfig => {
  const result = gatewayConfigFrom({ baseUrl, apiKey });
  if (result.kind !== "ok") {
    throw new Error(result.kind);
  }
  return result.config;
};

const sse = (response: ServerResponse, events: readonly object[]): void => {
  response.writeHead(200, { "content-type": "text/event-stream" });
  events.forEach((event) => response.write(`data: ${JSON.stringify(event)}\n\n`));
  response.end("data: [DONE]\n\n");
};

describe("gateway URLs", () => {
  it("accepts the gateway root with or without a trailing slash or /v1", () => {
    expect(gatewayRoot("https://litellm.example.com/")).toBe("https://litellm.example.com");
    expect(gatewayRoot("https://litellm.example.com/v1")).toBe("https://litellm.example.com");
    expect(gatewayRoot(" http://localhost:4000 ")).toBe("http://localhost:4000");
    expect(modelGroupInfoUrl("https://litellm.example.com")).toBe("https://litellm.example.com/model_group/info");
    expect(openAiBaseUrl("https://litellm.example.com")).toBe("https://litellm.example.com/v1");
  });

  it("rejects anything that is not an http or https URL", () => {
    expect(gatewayRoot("litellm.example.com")).toBeUndefined();
    expect(gatewayRoot("ftp://litellm.example.com")).toBeUndefined();
    expect(gatewayRoot("")).toBeUndefined();
  });
});

describe("gatewayConfigFrom", () => {
  it("distinguishes the unconfigured probe, a lost secret, and a bad URL from a usable configuration", () => {
    expect(gatewayConfigFrom(undefined)).toEqual({ kind: "unconfigured" });
    expect(gatewayConfigFrom({ baseUrl: "http://localhost:4000", apiKey: undefined })).toEqual({ kind: "missing_fields", fields: ["API key"] });
    expect(gatewayConfigFrom({ baseUrl: "  ", apiKey: "" })).toEqual({ kind: "missing_fields", fields: ["Gateway URL", "API key"] });
    expect(gatewayConfigFrom({ baseUrl: "localhost:4000", apiKey: "sk" })).toEqual({ kind: "invalid_url", baseUrl: "localhost:4000" });
    expect(gatewayConfigFrom({ baseUrl: " http://localhost:4000/v1/ ", apiKey: " sk-test " })).toEqual({
      kind: "ok",
      config: { baseUrl: "http://localhost:4000", apiKey: "sk-test" },
    });
  });
});

describe("summarizeErrorBody", () => {
  it("prefers the gateway's error message and caps the length", () => {
    expect(summarizeErrorBody('{"error":{"message":"invalid key","type":"auth_error","param":"sk-...abcd"}}')).toBe("invalid key");
    expect(summarizeErrorBody('{"detail":"Not Found"}')).toBe("Not Found");
    expect(summarizeErrorBody("<html>\n  502 Bad Gateway\n</html>")).toBe("<html> 502 Bad Gateway </html>");
    const long = summarizeErrorBody("x".repeat(ERROR_SUMMARY_LIMIT + 50));
    expect(long).toBe(`${"x".repeat(ERROR_SUMMARY_LIMIT)}...`);
  });
});

describe("listModelGroups", () => {
  it("calls /model_group/info with the virtual key and this extension's user agent", async () => {
    const gateway = await startGateway((_request, response) => {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ data: [{ model_group: "gpt-5.6", mode: "chat", input_cost_per_token: 4e-6 }] }));
    });
    const result = await createGatewayClient().listModelGroups(configFor(`${gateway.url}/v1`, "sk-test"), new AbortController().signal);
    expect(result).toEqual({
      kind: "ok",
      groups: [expect.objectContaining({ modelGroup: "gpt-5.6", inputCostPerToken: 4e-6 })],
    });
    expect(gateway.requests).toEqual([
      expect.objectContaining({ method: "GET", url: "/model_group/info", authorization: "Bearer sk-test", userAgent: USER_AGENT }),
    ]);
  });

  it("reports the gateway's status and body when the key is rejected", async () => {
    const gateway = await startGateway((_request, response) => {
      response.writeHead(401, { "content-type": "application/json" });
      response.end('{"error":{"message":"invalid key"}}');
    });
    expect(await createGatewayClient().listModelGroups({ baseUrl: gateway.url, apiKey: "sk-bad" }, new AbortController().signal)).toEqual({
      kind: "http_error",
      status: 401,
      body: '{"error":{"message":"invalid key"}}',
    });
  });

  it("reports a payload that is not a model group listing", async () => {
    const gateway = await startGateway((_request, response) => {
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"object":"list","models":[]}');
    });
    expect(await createGatewayClient().listModelGroups({ baseUrl: gateway.url, apiKey: "sk" }, new AbortController().signal)).toEqual({
      kind: "invalid_response",
      reason: "response has no data array",
    });
  });
});

describe("streamChatCompletion", () => {
  it("streams /v1/chat/completions through the gateway with the chosen reasoning effort", async () => {
    const gateway = await startGateway((_request, response) =>
      sse(response, [
        { id: "c", object: "chat.completion.chunk", created: 0, model: "gpt-5.6", choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: null }] },
        { id: "c", object: "chat.completion.chunk", created: 0, model: "gpt-5.6", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
      ]),
    );
    const params = buildChatCompletionParams({
      model: "gpt-5.6",
      messages: [{ role: 1, content: [{ value: "hello" }], name: undefined }],
      tools: [],
      requireToolCall: false,
      reasoningEffort: "high",
      modelOptions: {},
    });
    const chunks = await createGatewayClient().streamChatCompletion(configFor(`${gateway.url}/`, "sk-test"), params, new AbortController().signal);
    const contents: string[] = [];
    for await (const chunk of chunks) {
      contents.push(chunk.choices[0]?.delta.content ?? "");
    }
    expect(contents.join("")).toBe("Hi");
    const [request] = gateway.requests;
    expect(request).toMatchObject({ method: "POST", url: "/v1/chat/completions", authorization: "Bearer sk-test", userAgent: USER_AGENT });
    expect(JSON.parse(request?.body ?? "{}")).toMatchObject({
      model: "gpt-5.6",
      stream: true,
      stream_options: { include_usage: true },
      reasoning_effort: "high",
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
    });
  });

  it("leaves retries to the gateway instead of resending a failed request", async () => {
    const gateway = await startGateway((_request, response) => {
      response.writeHead(502, { "content-type": "application/json" });
      response.end('{"error":{"message":"upstream unavailable"}}');
    });
    const params = buildChatCompletionParams({
      model: "gpt-5.6",
      messages: [{ role: 1, content: [{ value: "hello" }], name: undefined }],
      tools: [],
      requireToolCall: false,
      reasoningEffort: undefined,
      modelOptions: {},
    });
    await expect(
      createGatewayClient().streamChatCompletion({ baseUrl: gateway.url, apiKey: "sk-test" }, params, new AbortController().signal),
    ).rejects.toThrow(/upstream unavailable/);
    expect(gateway.requests).toHaveLength(1);
  });
});
