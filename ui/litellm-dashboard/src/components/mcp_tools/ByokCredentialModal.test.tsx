import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { registerAuthHeaderNameGetter, registerAuthTokenGetter, registerBaseUrlGetter } from "@/lib/http/runtime";
import { ByokCredentialModal } from "./ByokCredentialModal";
import type { MCPServer } from "./types";

const fetchSpy = vi.hoisted(() => {
  const spy = vi.fn<(request: Request) => Promise<Response>>();
  vi.stubGlobal("fetch", spy);
  return spy;
});

vi.mock("@/components/molecules/message_manager", () => ({
  default: { success: vi.fn(), error: vi.fn() },
}));

const SERVER = { server_id: "srv-1", alias: "Linear", server_name: "Linear" } as MCPServer;

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

async function fillAndSubmit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText("Continue to Authentication"));
  await user.type(screen.getByPlaceholderText("Enter your API key"), "linear-key");
  await user.click(screen.getByRole("button", { name: /Connect & Authorize/ }));
}

beforeEach(() => {
  fetchSpy.mockReset();
  registerBaseUrlGetter(() => "");
  registerAuthTokenGetter(() => "sk-session");
});

describe("ByokCredentialModal", () => {
  it("saves the credential with the session's configured litellm key header, not a hardcoded Authorization", async () => {
    registerAuthHeaderNameGetter(() => "x-litellm-api-key");
    fetchSpy.mockResolvedValue(jsonResponse({ server_id: "srv-1", has_credential: true }));
    const onSuccess = vi.fn();
    const user = userEvent.setup();
    render(<ByokCredentialModal server={SERVER} open onClose={() => {}} onSuccess={onSuccess} />);

    await fillAndSubmit(user);

    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith("srv-1"));
    const request = fetchSpy.mock.calls[0][0];
    expect(request.method).toBe("POST");
    expect(new URL(request.url).pathname).toBe("/v1/mcp/server/srv-1/user-credential");
    expect(request.headers.get("x-litellm-api-key")).toBe("Bearer sk-session");
    expect(request.headers.get("Authorization")).toBeNull();
    expect(await request.json()).toEqual({ credential: "linear-key", save: true });
  });

  it("surfaces the backend's detail.error message when the save fails", async () => {
    registerAuthHeaderNameGetter(() => "Authorization");
    fetchSpy.mockResolvedValue(
      jsonResponse({ detail: { error: "This MCP server does not support BYOK credentials" } }, 400),
    );
    const MessageManager = (await import("@/components/molecules/message_manager")).default;
    const onSuccess = vi.fn();
    const user = userEvent.setup();
    render(<ByokCredentialModal server={SERVER} open onClose={() => {}} onSuccess={onSuccess} />);

    await fillAndSubmit(user);

    await waitFor(() =>
      expect(MessageManager.error).toHaveBeenCalledWith("This MCP server does not support BYOK credentials"),
    );
    expect(onSuccess).not.toHaveBeenCalled();
  });
});
