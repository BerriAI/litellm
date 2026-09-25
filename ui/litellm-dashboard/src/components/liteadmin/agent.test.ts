// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import {
  MAX_INPUT_LENGTH,
  resolveInferenceTarget,
  runLiteAdmin,
  type LiteAdminOptions,
  type AgentSocket,
} from "./agent";

class Socket {
  onopen: AgentSocket["onopen"] = null;
  onmessage: AgentSocket["onmessage"] = null;
  onerror: AgentSocket["onerror"] = null;
  onclose: AgentSocket["onclose"] = null;
  send = vi.fn();
  close = vi.fn();
  open() {
    this.onopen?.call(this as unknown as WebSocket, new Event("open"));
  }
  event(data: unknown) {
    return this.onmessage?.call(
      this as unknown as WebSocket,
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }
  disconnect() {
    this.onclose?.call(this as unknown as WebSocket, new CloseEvent("close"));
  }
}

function options(): LiteAdminOptions {
  return {
    accessToken: "test-session",
    managementBaseUrl: "https://gateway.test/proxy",
    inferenceBaseUrl: "https://models.test",
    model: "chosen-model",
    messages: [{ role: "user", content: "Check budget" }],
    signal: new AbortController().signal,
    assertCurrent: vi.fn(),
    onMessage: vi.fn(),
    confirm: vi.fn(async () => true),
    onResult: vi.fn(),
  };
}
const action = {
  id: "action-1",
  name: "team_update",
  title: "Update a team",
  arguments: { team_id: "team-1", max_budget: 50 },
  destructive: false,
};

it("sends the session and selected model over the management WebSocket and renders backend answers", async () => {
  const socket = new Socket();
  const connect = vi.fn(() => socket);
  const input = options();
  const running = runLiteAdmin(input, connect);
  socket.open();
  expect(connect).toHaveBeenCalledWith("wss://gateway.test/proxy/liteadmin/chat");
  expect(JSON.parse(socket.send.mock.calls[0][0])).toEqual({
    access_token: "test-session",
    chat: {
      model: "chosen-model",
      messages: [{ role: "user", content: "Check budget" }],
      inference_base_url: "https://models.test",
    },
  });
  await socket.event({ type: "message", text: "The budget is $50." });
  await socket.event({ type: "done" });
  await running;
  expect(input.onMessage).toHaveBeenCalledWith("The budget is $50.");
  expect(socket.close).toHaveBeenCalledOnce();
});

it("waits for the user's decision before sending approval and keeps generated keys in action results", async () => {
  const socket = new Socket();
  let decide!: (value: boolean) => void;
  const decision = new Promise<boolean>((resolve) => {
    decide = resolve;
  });
  const input = { ...options(), confirm: vi.fn(() => decision) };
  const running = runLiteAdmin(input, () => socket);
  socket.open();
  const received = socket.event({ type: "approval", action });
  expect(input.confirm).toHaveBeenCalledWith(action);
  expect(socket.send).toHaveBeenCalledTimes(1);
  decide(true);
  await received;
  expect(JSON.parse(socket.send.mock.calls[1][0])).toEqual({ id: action.id, approved: true });
  await socket.event({ type: "result", action, result: { status: "completed", key: "sk-new-key" } });
  await socket.event({ type: "done" });
  await running;
  expect(input.onResult).toHaveBeenCalledWith(action, { status: "completed", key: "sk-new-key" });
  expect(input.onMessage).not.toHaveBeenCalled();
});

it("reports an uncertain outcome when a submitted change loses its connection", async () => {
  const socket = new Socket();
  const input = options();
  const running = runLiteAdmin(input, () => socket);
  const failed = expect(running).rejects.toThrow("connection closed");
  socket.open();
  await socket.event({ type: "approval", action });
  socket.disconnect();
  await failed;
  expect(input.onResult).toHaveBeenCalledWith(action, {
    status: "unknown",
    message: expect.stringContaining("Check the resource"),
  });
});

it("cancels without approving when the session changes while reviewing", async () => {
  const socket = new Socket();
  const input = options();
  input.confirm = vi.fn(async () => {
    input.assertCurrent = () => {
      throw new Error("Session changed");
    };
    return true;
  });
  const running = runLiteAdmin(input, () => socket);
  const failed = expect(running).rejects.toThrow("Session changed");
  socket.open();
  await socket.event({ type: "approval", action });
  await failed;
  expect(socket.send).toHaveBeenCalledTimes(1);
});

it("closes the backend connection when the request is aborted", async () => {
  const socket = new Socket();
  const controller = new AbortController();
  const running = runLiteAdmin({ ...options(), signal: controller.signal }, () => socket);
  const failed = expect(running).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await failed;
  expect(socket.close).toHaveBeenCalledOnce();
  expect(socket.send).not.toHaveBeenCalled();
});

it("never opens a connection for an oversized message", async () => {
  const connect = vi.fn();
  await expect(
    runLiteAdmin({ ...options(), messages: [{ role: "user", content: "x".repeat(MAX_INPUT_LENGTH + 1) }] }, connect),
  ).rejects.toThrow("8,000");
  expect(connect).not.toHaveBeenCalled();
});
describe("inference destination", () => {
  const destinations = [
    ["", "/gateway", "https://dashboard.example/ui/", "https://dashboard.example/gateway", false],
    ["v1", "/gateway", "https://dashboard.example/ui/", "https://dashboard.example/v1", false],
    [
      "https://DASHBOARD.example:443/gateway",
      "/gateway",
      "https://dashboard.example/ui/",
      "https://dashboard.example/gateway",
      false,
    ],
    ["https://models.example/v1", "/gateway", "https://dashboard.example/ui/", "https://models.example/v1", true],
    ["//models.example/v1", "/gateway", "https://dashboard.example/ui/", "https://models.example/v1", true],
    ["", "https://management.example/root", "https://dashboard.example/ui/", "https://management.example/root", false],
    ["http://localhost:4001", "http://localhost:4000", "http://localhost:3000/ui", "http://localhost:4001/", true],
  ] as const;
  const destinationCases = destinations.map(([candidate, management, page, baseUrl, requiresConsent]) => ({
    candidate,
    management,
    page,
    baseUrl,
    requiresConsent,
  }));
  it.each(destinationCases)(
    "resolves $candidate and requires consent only for a distinct management origin",
    ({ candidate, management, page, baseUrl, requiresConsent }) => {
      expect(resolveInferenceTarget(candidate, management, page)).toEqual({ baseUrl, requiresConsent, error: null });
    },
  );

  it.each([
    "https://",
    "javascript:alert(1)",
    "https://user:secret@models.example",
    "https://models.example?secret",
    "https://models.example#secret",
    "https://models.example?",
    "https://models.example#",
    "http://models.example",
  ])("rejects unsafe inference configuration without echoing secrets", (candidate) => {
    const result = resolveInferenceTarget(candidate, "https://management.example", "https://dashboard.example/ui/");
    expect(result).toMatchObject({ baseUrl: null, requiresConsent: false, error: expect.any(String) });
    expect(result.error).not.toContain("secret");
  });

  it("rejects an HTTPS management downgrade even on an HTTP development page", () => {
    expect(
      resolveInferenceTarget("http://models.example", "https://management.example", "http://localhost/ui").baseUrl,
    ).toBeNull();
  });
});
