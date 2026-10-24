import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import ChatConversationPage from "./page";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));

const teams = [
  { team_id: "research", team_alias: "Research" },
  { team_id: "engineering", team_alias: "Engineering" },
];
const requests: { path: string; teamId: string | null; body: Record<string, unknown> }[] = [];
let models: ModelGroup[];

function renderChat() {
  return renderWithProviders(
    <ChatShellProvider
      accessToken="session-token"
      userId="team-user"
      userEmail="member@example.com"
      userRole="Internal User"
      premiumUser={false}
    >
      <ChatConversationPage />
    </ChatShellProvider>,
  );
}

function enterPrompt(text = "Hello") {
  act(() =>
    fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), { target: { value: text } }),
  );
}

beforeEach(() => {
  localStorage.clear();
  testQueryClient.clear();
  requests.length = 0;
  models = [{ model_group: "shared-chat", mode: "chat", providers: ["openai"], available_teams: teams }];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request =
        input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
      const path = new URL(request.url).pathname;
      if (path.endsWith("/model_group/info")) return Response.json({ data: models });
      if (path.endsWith("/responses") || path.endsWith("/images/generations")) {
        requests.push({
          path,
          teamId: request.headers.get("x-litellm-team-id"),
          body: JSON.parse(await request.text()),
        });
        if (path.endsWith("/images/generations"))
          return Response.json({ created: 1, data: [{ b64_json: "aGVsbG8=" }] });
        const events = [
          { type: "response.output_text.delta", delta: `Reply ${requests.length}` },
          { type: "response.completed", response: { id: `response-${requests.length}`, output: [] } },
        ];
        return new Response(
          events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""),
          { headers: { "Content-Type": "text/event-stream" } },
        );
      }
      return Response.json({}, { status: 404 });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Chat session team selection", () => {
  it("should render the model and require a billing choice only when multiple teams grant access", async () => {
    renderChat();
    await screen.findByRole("button", { name: "shared-chat" });
    expect(screen.getByRole("combobox", { name: "Team paying for this request" })).toHaveTextContent("Choose team");
    enterPrompt();
    expect(screen.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
    expect(requests).toHaveLength(0);
  });

  it.each([
    { allowed: [], teamId: null },
    { allowed: [teams[0]], teamId: "research" },
  ])("should submit without a team selector when access is unambiguous ($teamId)", async ({ allowed, teamId }) => {
    models[0].available_teams = allowed;
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "shared-chat" });
    expect(screen.queryByRole("combobox", { name: "Team paying for this request" })).not.toBeInTheDocument();
    enterPrompt();
    await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    await screen.findByText("Reply 1");
    expect(requests[0].teamId).toBe(teamId);
  });

  it("should charge the selected team and replay history when switching teams", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "shared-chat" });
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Research" }));
    enterPrompt("First question");
    await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    await screen.findByText("Reply 1");
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    enterPrompt("Follow up");
    await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    await screen.findByText("Reply 2");
    expect(requests.map((request) => request.teamId)).toEqual(["research", "engineering"]);
    expect(requests[1].body).not.toHaveProperty("previous_response_id");
    expect(requests[1].body.input).toEqual(
      expect.arrayContaining([expect.objectContaining({ role: "user", content: "First question" })]),
    );
  });

  it("should send image generation to the chosen team and clear the choice when changing models", async () => {
    models.push({ model_group: "studio-image", mode: "image_generation", available_teams: teams });
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "shared-chat" });
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Research" }));
    await user.click(screen.getByRole("button", { name: "shared-chat", exact: true }));
    await user.click(await screen.findByRole("button", { name: /^studio-image/ }));
    enterPrompt("Draw a circle");
    expect(screen.getByRole("button", { name: "Generate image", exact: true })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    await user.click(screen.getByRole("button", { name: "Generate image", exact: true }));
    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toMatchObject({
      teamId: "engineering",
      body: { model: "studio-image", prompt: "Draw a circle" },
    });
  });

  it("should retain uploaded documents and replay them under the newly selected team", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "shared-chat" });
    const file = new File(["team document"], "brief.txt", { type: "text/plain" });
    await user.upload(screen.getByLabelText("Upload files"), file);
    await screen.findByRole("button", { name: "Remove brief.txt" });
    enterPrompt("Read the file");
    expect(screen.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Research" }));
    await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    await screen.findByText("Reply 1");
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    enterPrompt("Use that file again");
    await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    await screen.findByText("Reply 2");

    const document = {
      type: "input_file",
      filename: "brief.txt",
      file_data: "data:text/plain;base64,dGVhbSBkb2N1bWVudA==",
    };
    const originalMessage = expect.objectContaining({
      role: "user",
      content: expect.arrayContaining([document]),
    });
    expect(requests.map((request) => request.teamId)).toEqual(["research", "engineering"]);
    expect(requests[0].body.input).toEqual(expect.arrayContaining([originalMessage]));
    expect(requests[1].body).not.toHaveProperty("previous_response_id");
    expect(requests[1].body.input).toEqual(expect.arrayContaining([originalMessage]));
  });
});
