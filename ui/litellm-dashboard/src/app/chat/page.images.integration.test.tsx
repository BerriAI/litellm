import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import { recordChatQuery } from "@/lib/telemetry/chat";
import ChatConversationPage from "./page";

vi.mock("@/lib/telemetry/chat", () => ({ recordChatQuery: vi.fn(async () => {}) }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

type RecordedRequest = { pathname: string; body: Record<string, unknown> };
const requests: RecordedRequest[] = [];
const pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZQAAAAASUVORK5CYII=";
let models: { model_group: string; mode: string; providers: string[] }[];
let imageResponse: () => Response;
let pendingGeneration: boolean;
let generationSignal: AbortSignal | null;
let includeMCPEvents: boolean;

function responseStream() {
  return new Response(
    [
      { type: "response.output_text.delta", delta: "I can see " },
      ...(includeMCPEvents
        ? ["search_notes", "read_note"].map((name) => ({
            type: "response.output_item.done",
            item: { id: name, type: "mcp_call", name, server_label: "research-tools", output: "Found a note" },
          }))
        : []),
      { type: "response.output_text.delta", delta: "the uploaded image." },
      { type: "response.completed", response: { id: `response-${requests.length}`, output: [] } },
    ]
      .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join(""),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

function renderChat() {
  return renderWithProviders(
    <ChatShellProvider
      accessToken="sk-media-test"
      userId="media-test-user"
      userEmail="researcher@example.com"
      userRole="Internal Viewer"
      premiumUser={false}
    >
      <ChatConversationPage />
    </ChatShellProvider>,
  );
}

async function uploadImage(user: ReturnType<typeof userEvent.setup>, name = "diagram.png") {
  const bytes = Uint8Array.from(atob(pixel), (character) => character.charCodeAt(0));
  await user.upload(screen.getByLabelText("Upload files"), new File([bytes], name, { type: "image/png" }));
  await screen.findByRole("img", { name });
}

async function chooseModel(user: ReturnType<typeof userEvent.setup>, current: string, next: string) {
  await user.click(screen.getByRole("button", { name: current, exact: true }));
  await user.click(await screen.findByRole("button", { name: new RegExp(`^${next}`) }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "All", exact: true })).not.toBeInTheDocument());
}

beforeEach(() => {
  localStorage.clear();
  testQueryClient.clear();
  requests.length = 0;
  vi.mocked(recordChatQuery).mockClear();
  models = [
    { model_group: "research-chat", mode: "chat", providers: ["openai"] },
    { model_group: "studio-image", mode: "image_generation", providers: ["openai"] },
  ];
  imageResponse = () => Response.json({ created: 1, data: [{ b64_json: pixel }] });
  pendingGeneration = false;
  generationSignal = null;
  includeMCPEvents = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request =
        input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
      const pathname = new URL(request.url).pathname;
      if (pathname.endsWith("/model_group/info")) return Response.json({ data: models });
      if (pathname.endsWith("/v1/mcp/server")) {
        return Response.json([{ server_id: "research-tools", server_name: "research-tools" }]);
      }
      if (pathname.endsWith("/mcp-rest/tools/list")) return Response.json({ tools: [] });
      if (pathname.endsWith("/responses") || pathname.endsWith("/images/generations")) {
        requests.push({ pathname, body: JSON.parse(await request.text()) });
        if (pathname.endsWith("/responses")) return responseStream();
        if (pendingGeneration) {
          generationSignal = request.signal;
          return new Promise<Response>((_resolve, reject) => {
            request.signal.addEventListener("abort", () => reject(new DOMException("Stopped", "AbortError")), {
              once: true,
            });
          });
        }
        return imageResponse();
      }
      return Response.json({}, { status: 404 });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("chat image workflows", () => {
  it("should render the permitted models and image controls for a research team member", async () => {
    renderChat();
    expect(await screen.findByRole("button", { name: "research-chat" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Attach files" })).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "research-chat", exact: true }));
    expect(await screen.findByRole("button", { name: /^studio-image/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(recordChatQuery).not.toHaveBeenCalled();
  });

  it("should upload a document as a file part, preserve it on edit, and keep contents out of metrics", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await user.upload(
      screen.getByLabelText("Upload files"),
      new File(["private fixture"], "brief.docx", {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    );
    expect(await screen.findByText("brief.docx")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "brief.docx" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("I can see the uploaded image.");
    const filePart = {
      type: "input_file",
      filename: "brief.docx",
      file_data:
        "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,cHJpdmF0ZSBmaXh0dXJl",
    };
    expect(requests[0].body).toMatchObject({ input: [{ role: "user", content: [filePart] }] });
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
    await user.click(screen.getByRole("button", { name: "Edit message" }));
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Edit message" }), {
        target: { value: "Summarize the document" },
      }),
    );
    await user.click(screen.getByRole("button", { name: "Save & Send" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1].body).toMatchObject({
      input: [{ role: "user", content: [{ type: "input_text", text: "Summarize the document" }, filePart] }],
    });
    expect(requests[1].body).not.toHaveProperty("previous_response_id");
  });

  it("should enforce the shared file limit and remove a document before submission", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    const files = Array.from(
      { length: 5 },
      (_, index) => new File(["text"], `note-${index}.txt`, { type: "text/plain" }),
    );
    await user.upload(screen.getByLabelText("Upload files"), files);
    expect(await screen.findByRole("alert")).toHaveTextContent("Attach up to 4 files per message.");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await user.upload(screen.getByLabelText("Upload files"), files[0]);
    await user.click(await screen.findByRole("button", { name: "Remove note-0.txt" }));
    expect(screen.queryByText("note-0.txt")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(recordChatQuery).not.toHaveBeenCalled();
  });

  it("should restore the selected chat model after switching modes and reopening the page", async () => {
    const user = userEvent.setup();
    const view = renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await chooseModel(user, "research-chat", "studio-image");
    expect(screen.getByRole("button", { name: "studio-image" })).toBeInTheDocument();
    await chooseModel(user, "studio-image", "research-chat");
    view.unmount();
    renderChat();
    expect(await screen.findByRole("button", { name: "research-chat" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Message", exact: true })).toBeInTheDocument();
  });

  it("should preview and remove an uploaded image before sending", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await uploadImage(user);
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Remove diagram.png" }));
    expect(screen.queryByRole("img", { name: "diagram.png" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(requests).toHaveLength(0);
    expect(recordChatQuery).not.toHaveBeenCalled();
  });

  it("should send the image and preserve it when editing and resending the message", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await uploadImage(user);
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "Explain this diagram");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("I can see the uploaded image.");
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
    expect(requests[0].body).toMatchObject({
      model: "research-chat",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Explain this diagram" },
            { type: "input_image", image_url: `data:image/png;base64,${pixel}` },
          ],
        },
      ],
    });
    expect(screen.getByRole("img", { name: "diagram.png" })).toBeInTheDocument();
    await user.hover(screen.getAllByText("Explain this diagram").at(-1)!);
    const editButton = await screen.findByRole("button", { name: "Edit message" });
    act(() => fireEvent.click(editButton));
    await user.clear(screen.getByRole("textbox", { name: "Edit message" }));
    await user.type(screen.getByRole("textbox", { name: "Edit message" }), "Explain the arrows");
    await user.click(screen.getByRole("button", { name: "Save & Send" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1].body).toMatchObject({
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Explain the arrows" },
            { type: "input_image", image_url: `data:image/png;base64,${pixel}` },
          ],
        },
      ],
    });
    expect(requests[1].body).not.toHaveProperty("previous_response_id");
    expect(screen.queryByText("Explain this diagram", { selector: "p" })).not.toBeInTheDocument();
    expect(recordChatQuery).toHaveBeenCalledTimes(2);
  });

  it("should preserve a chat turn when editing it while an image model is selected", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "Explain the experiment");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("I can see the uploaded image.");
    await chooseModel(user, "research-chat", "studio-image");
    await user.hover(screen.getAllByText("Explain the experiment").at(-1)!);
    const editButton = await screen.findByRole("button", { name: "Edit message" });
    act(() => fireEvent.click(editButton));
    await user.clear(screen.getByRole("textbox", { name: "Edit message" }));
    await user.type(screen.getByRole("textbox", { name: "Edit message" }), "Explain the findings");
    await user.click(screen.getByRole("button", { name: "Save & Send" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose a chat model before resending this message.");
    expect(screen.getByRole("heading", { name: "Explain the experiment" })).toBeInTheDocument();
    expect(screen.getByText("I can see the uploaded image.")).toBeInTheDocument();
    expect(requests).toHaveLength(1);
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
  });

  it("should continue an image conversation using the completed response context", async () => {
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await uploadImage(user);
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("I can see the uploaded image.");
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "What is the main color?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1].body).toMatchObject({
      previous_response_id: "response-1",
      input: [{ role: "user", content: "What is the main color?" }],
    });
    expect(screen.getByRole("img", { name: "diagram.png" })).toBeInTheDocument();
  });

  it("should generate and render an image through the selected Model Hub model", async () => {
    vi.mocked(recordChatQuery).mockReturnValueOnce(new Promise<void>(() => {}));
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await chooseModel(user, "research-chat", "studio-image");
    expect(screen.getByRole("button", { name: "studio-image" })).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "A blue research lab");
    await user.click(screen.getByRole("button", { name: "Generate image" }));
    const result = await screen.findByRole("img", { name: "Generated image 1" });
    expect(result).toHaveAttribute("src", `data:image/png;base64,${pixel}`);
    expect(requests).toEqual([
      { pathname: "/images/generations", body: { model: "studio-image", prompt: "A blue research lab" } },
    ]);
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
  });

  it("should list no image models when the account has none permitted", async () => {
    models = models.filter((model) => model.mode === "chat");
    const user = userEvent.setup();
    renderChat();
    await user.click(await screen.findByRole("button", { name: "research-chat" }));
    await user.click(screen.getByRole("button", { name: "Images", exact: true }));
    expect(screen.getByText("No models found.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^studio-image/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate image" })).not.toBeInTheDocument();
    expect(requests).toHaveLength(0);
  });

  it("should show the provider error when image generation fails", async () => {
    imageResponse = () =>
      Response.json({ error: { message: "Image access is disabled for this key" } }, { status: 403 });
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await chooseModel(user, "research-chat", "studio-image");
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "A blue lab");
    await user.click(screen.getByRole("button", { name: "Generate image" }));
    expect(await screen.findByText(/Image access is disabled for this key/)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Generated image 1" })).not.toBeInTheDocument();
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
  });

  it("should abort image generation when the user stops the request", async () => {
    pendingGeneration = true;
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await chooseModel(user, "research-chat", "studio-image");
    await user.type(screen.getByRole("textbox", { name: "Message", exact: true }), "A blue lab");
    await user.click(screen.getByRole("button", { name: "Generate image" }));
    await waitFor(() => expect(generationSignal).not.toBeNull());
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
        target: { value: "Another image" },
      });
      fireEvent.keyDown(screen.getByRole("textbox", { name: "Message", exact: true }), { key: "Enter" });
    });
    expect(requests).toHaveLength(1);
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
    await user.click(screen.getByRole("button", { name: "Stop request" }));
    expect(generationSignal?.aborted).toBe(true);
    expect(await screen.findByText("[stopped]")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Generated image 1" })).not.toBeInTheDocument();
    expect(recordChatQuery).toHaveBeenCalledTimes(1);
  });

  it("should reject unsupported pasted files without issuing a request", async () => {
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    act(() => {
      fireEvent.paste(screen.getByRole("textbox", { name: "Message", exact: true }), {
        clipboardData: { files: [new File(["svg"], "diagram.svg", { type: "image/svg+xml" })] },
      });
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose a PNG, JPEG, WebP, or GIF image.");
    expect(requests).toHaveLength(0);
    expect(recordChatQuery).not.toHaveBeenCalled();
  });

  it("should not count a blank submission as a query", async () => {
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
        target: { value: "   " },
      });
      fireEvent.keyDown(screen.getByRole("textbox", { name: "Message", exact: true }), { key: "Enter" });
    });
    expect(requests).toHaveLength(0);
    expect(recordChatQuery).not.toHaveBeenCalled();
  });

  it("should count one query for a message that streams text and executes multiple MCP tools", async () => {
    includeMCPEvents = true;
    const user = userEvent.setup();
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await user.click(screen.getByRole("button", { name: "Connect tools" }));
    const serverSwitch = await screen.findByRole("switch");
    await user.click(serverSwitch);
    await waitFor(() => expect(screen.getByRole("switch")).toBeChecked());
    await user.click(screen.getByRole("button", { name: "Connect tools" }));
    expect(recordChatQuery).not.toHaveBeenCalled();

    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
        target: { value: "Search the research notes" },
      });
    });
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("I can see the uploaded image.");

    expect(requests).toHaveLength(1);
    expect(requests[0].body).toMatchObject({
      tools: [{ type: "mcp", server_label: "research-tools", require_approval: "never" }],
    });
    expect(recordChatQuery).toHaveBeenCalledExactlyOnceWith("media-test-user");
  });
});
