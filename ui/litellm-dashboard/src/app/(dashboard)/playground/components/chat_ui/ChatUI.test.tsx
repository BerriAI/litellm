import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders as render } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatUI from "./ChatUI";
import * as fetchModelsModule from "@/components/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "@/components/llm_calls/chat_completion";

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

vi.mock("@/components/llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn().mockResolvedValue({}),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ data: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
  fetchMCPServers: vi.fn().mockResolvedValue([]),
  fetchMCPToolsets: vi.fn().mockResolvedValue([]),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  callMCPTool: vi.fn(),
}));

beforeEach(() => {
  Element.prototype.scrollIntoView = () => {};
});

const CHAT_REQUEST_ARG_COUNT = 26;
const STREAMING_ENABLED_ARG_INDEX = 25;

async function openComboboxByPlaceholder(placeholder: string) {
  const user = userEvent.setup();
  const combobox = await screen.findByPlaceholderText(placeholder);
  await user.click(combobox);
  return combobox;
}

async function selectComboboxOption(placeholder: string, optionLabel: string) {
  const user = userEvent.setup();
  await openComboboxByPlaceholder(placeholder);
  const option = await screen.findByText(optionLabel);
  await user.click(option);
}

describe("ChatUI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    Element.prototype.scrollIntoView = vi.fn();

    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValue([
      { model_group: "Model 1", mode: "chat" },
      { model_group: "Model 2", mode: "chat" },
      { model_group: "Model 3", mode: "chat" },
    ]);
  });

  it("should render the chat UI", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );
    expect(screen.getByText("Test Key")).toBeInTheDocument();
  });

  it("should show the voice selector when the endpoint type is audio_speech", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/audio/speech");

    await waitFor(() => {
      expect(screen.getByText("Voice")).toBeInTheDocument();
      expect(screen.getByLabelText("Voice")).toBeInTheDocument();
    });
  });

  it("should show the SDK type by its human label rather than its wire value", async () => {
    const user = userEvent.setup();
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /get code/i }));

    const sdkTrigger = await screen.findByLabelText("SDK Type");
    expect(sdkTrigger).toHaveTextContent("OpenAI SDK");

    await user.click(sdkTrigger);
    await user.click(await screen.findByRole("option", { name: "Azure SDK" }));

    expect(await screen.findByLabelText("SDK Type")).toHaveTextContent("Azure SDK");
  });

  it("should show the voice by its human label rather than its wire value", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/audio/speech");

    await waitFor(() => {
      expect(screen.getByLabelText("Voice")).toHaveTextContent("Alloy - Professional and confident");
    });
  });

  it("should allow the user to select a model", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await openComboboxByPlaceholder("Select a Model");

    await waitFor(() => {
      expect(screen.getAllByText("Model 1").length).toBeGreaterThan(0);
    });
  });

  it("shows only endpoint-compatible models when chat endpoint is selected", async () => {
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { model_group: "ChatModel", mode: "chat" },
      { model_group: "SpeechModel", mode: "audio_speech" },
      { model_group: "ImageModel", mode: "image_generation" },
      { model_group: "ResponsesModel", mode: "responses" },
      { model_group: "RealtimeModel", mode: "realtime" },
      { model_group: "NoModeModel" },
    ]);

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/chat/completions");
    await openComboboxByPlaceholder("Select a Model");

    await waitFor(() => {
      expect(screen.getAllByText("ChatModel").length).toBeGreaterThan(0);
      expect(screen.getAllByText("NoModeModel").length).toBeGreaterThan(0);
      expect(screen.queryByText("SpeechModel")).not.toBeInTheDocument();
      expect(screen.queryByText("ImageModel")).not.toBeInTheDocument();
      expect(screen.queryByText("ResponsesModel")).not.toBeInTheDocument();
      expect(screen.queryByText("RealtimeModel")).not.toBeInTheDocument();
    });
  });

  it("shows only realtime models when realtime endpoint is selected", async () => {
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { model_group: "ChatModel", mode: "chat" },
      { model_group: "RealtimeModel", mode: "realtime" },
      { model_group: "NoModeModel" },
    ]);

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/realtime");
    await openComboboxByPlaceholder("Select a Model");

    await waitFor(() => {
      expect(screen.getAllByText("RealtimeModel").length).toBeGreaterThan(0);
      expect(screen.getAllByText("NoModeModel").length).toBeGreaterThan(0);
      expect(screen.queryByText("ChatModel")).not.toBeInTheDocument();
    });
  });

  it("should show 'Enter custom model' option in model selector", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await openComboboxByPlaceholder("Select a Model");

    await waitFor(() => {
      expect(screen.getByText("Enter custom model")).toBeInTheDocument();
    });
  });

  it("should enable the MCP tools selector for chat completions", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    const mcpInput = () => screen.getByLabelText("Select MCP servers");

    await selectComboboxOption("Select an endpoint", "/v1/embeddings");

    await waitFor(() => {
      expect(mcpInput()).toBeDisabled();
    });

    await selectComboboxOption("Select an endpoint", "/v1/chat/completions");

    await waitFor(() => {
      expect(mcpInput()).toBeEnabled();
    });
  });

  it("should show Simulate failure to test fallbacks in Model Settings when chat endpoint is selected", async () => {
    const user = userEvent.setup();
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select a Model", "Model 1");

    await waitFor(() => {
      expect(screen.getByTestId("model-settings-button")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("model-settings-button"));

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
      expect(screen.getByText(/Simulate failure to test fallbacks/i)).toBeInTheDocument();
    });

    const fallbacksCheckbox = screen.getByRole("checkbox", {
      name: /Simulate failure to test fallbacks/i,
    });
    expect(fallbacksCheckbox).not.toBeChecked();

    await user.click(fallbacksCheckbox);

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: /Simulate failure to test fallbacks/i })).toBeChecked();
    });
  });

  it("should send the chat request non-streaming after Stream responses is unchecked", async () => {
    const user = userEvent.setup();
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select a Model", "Model 1");

    await waitFor(() => {
      expect(screen.getByTestId("model-settings-button")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("model-settings-button"));

    const streamingCheckbox = await screen.findByRole("checkbox", { name: /Stream responses/i });
    expect(streamingCheckbox).toBeChecked();

    await user.click(streamingCheckbox);

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: /Stream responses/i })).not.toBeChecked();
    });

    const messageInput = screen.getByPlaceholderText("Type your message... (Shift+Enter for new line)");
    await act(async () => {
      fireEvent.change(messageInput, { target: { value: "hello" } });
    });
    await act(async () => {
      fireEvent.keyDown(messageInput, { key: "Enter", code: "Enter" });
    });

    await waitFor(() => {
      expect(makeOpenAIChatCompletionRequest).toHaveBeenCalledTimes(1);
    });

    const requestArgs = vi.mocked(makeOpenAIChatCompletionRequest).mock.calls[0];
    expect(requestArgs).toHaveLength(CHAT_REQUEST_ARG_COUNT);
    expect(requestArgs[STREAMING_ENABLED_ARG_INDEX]).toBe(false);
  });

  it("should force streaming in simplified mode even when the playground setting is off", async () => {
    sessionStorage.setItem("streamingEnabled", "false");

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
        simplified
        fixedModel="Model 1"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Chat")).toBeInTheDocument();
    });

    const messageInput = screen.getByPlaceholderText("Type your message... (Shift+Enter for new line)");
    await act(async () => {
      fireEvent.change(messageInput, { target: { value: "hello" } });
    });
    await act(async () => {
      fireEvent.keyDown(messageInput, { key: "Enter", code: "Enter" });
    });

    await waitFor(() => {
      expect(makeOpenAIChatCompletionRequest).toHaveBeenCalledTimes(1);
    });

    const requestArgs = vi.mocked(makeOpenAIChatCompletionRequest).mock.calls[0];
    expect(requestArgs).toHaveLength(CHAT_REQUEST_ARG_COUNT);
    expect(requestArgs[STREAMING_ENABLED_ARG_INDEX]).toBe(true);
    expect(sessionStorage.getItem("streamingEnabled")).toBe("false");
  });

  it("should offer the streaming toggle for a responses-only model without advanced params", async () => {
    const user = userEvent.setup();
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValue([
      { model_group: "ResponsesModel", mode: "responses" },
    ]);

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/responses");
    await selectComboboxOption("Select a Model", "ResponsesModel");

    await waitFor(() => {
      expect(screen.getByTestId("model-settings-button")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("model-settings-button"));

    expect(await screen.findByRole("checkbox", { name: /Stream responses/i })).toBeChecked();
    expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
    expect(screen.queryByText("Use Advanced Parameters")).not.toBeInTheDocument();
  });

  it("should show Fill button and populate customProxyBaseUrl when proxySettings.LITELLM_UI_API_DOC_BASE_URL is provided", async () => {
    const testProxyUrl = "http://localhost:5000";

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
        proxySettings={{
          LITELLM_UI_API_DOC_BASE_URL: testProxyUrl,
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    const fillButton = screen.getByText("Fill");
    expect(fillButton).toBeInTheDocument();

    act(() => {
      fireEvent.click(fillButton);
    });

    await waitFor(() => {
      expect(sessionStorage.getItem("customProxyBaseUrl")).toBe(testProxyUrl);
    });

    await waitFor(() => {
      expect(screen.queryByText("Fill")).not.toBeInTheDocument();
    });

    const customProxyInput = screen.getByPlaceholderText(
      "Optional: Enter custom proxy URL (e.g., http://localhost:5000)",
    );
    expect(customProxyInput).toHaveValue(testProxyUrl);
  });

  it("should enable search functionality for MCP server selector", async () => {
    const user = userEvent.setup();
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    expect(screen.getByText("MCP Servers")).toBeInTheDocument();

    const mcpInput = screen.getByLabelText("Select MCP servers");
    expect(mcpInput).toBeInTheDocument();
    expect(mcpInput).toBeEnabled();

    await user.click(mcpInput);

    await waitFor(() => {
      expect(screen.getByText("All MCP Servers")).toBeInTheDocument();
    });
  });

  it("should keep the chosen endpoint when a model that endpoint can serve is picked", async () => {
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { model_group: "ChatModel", mode: "chat" },
    ]);

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/responses");
    await selectComboboxOption("Select a Model", "ChatModel");

    expect(screen.getByPlaceholderText("Select an endpoint")).toHaveValue("/v1/responses");
  });

  it("should not offer a model the selected endpoint cannot serve", async () => {
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { model_group: "ChatModel", mode: "chat" },
      { model_group: "SpeechModel", mode: "audio_speech" },
    ]);

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/responses");
    await openComboboxByPlaceholder("Select a Model");

    await waitFor(() => {
      expect(screen.getAllByText("ChatModel").length).toBeGreaterThan(0);
    });
    expect(screen.queryByText("SpeechModel")).not.toBeInTheDocument();
  });

  it("should attach an audio file dropped on the transcription upload area", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await selectComboboxOption("Select an endpoint", "/v1/audio/transcriptions");

    const dropZone = (await screen.findByText("Click or drag audio file to upload")).closest("label");
    const file = new File(["clip"], "clip.wav", { type: "audio/wav" });
    fireEvent.drop(dropZone as HTMLElement, { dataTransfer: { files: [file] } });

    expect(await screen.findByText("clip.wav")).toBeInTheDocument();
  });

  it("should name the virtual key source options instead of showing raw values", async () => {
    const user = userEvent.setup();

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    const keySourceTrigger = screen.getByLabelText("Virtual Key Source");
    expect(keySourceTrigger).toHaveTextContent("Current UI Session");
    expect(keySourceTrigger).not.toHaveTextContent("session");

    await user.click(keySourceTrigger);
    await user.click(await screen.findByRole("option", { name: "Virtual Key" }));

    await waitFor(() => {
      expect(screen.getByLabelText("Virtual Key Source")).toHaveTextContent("Virtual Key");
    });
    expect(screen.getByLabelText("Virtual Key Source")).not.toHaveTextContent("custom");
  });

  it("should re-enable the model selector when the virtual key is cleared mid-load", async () => {
    const user = userEvent.setup();
    (fetchModelsModule.fetchAvailableModels as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise(() => {}),
    );

    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    await user.click(screen.getByLabelText("Virtual Key Source"));
    await user.click(await screen.findByRole("option", { name: "Virtual Key" }));

    const keyField = await screen.findByPlaceholderText("Enter custom Virtual Key");
    fireEvent.change(keyField, { target: { value: "sk-test" } });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Loading models...")).toBeInTheDocument();
    });

    await user.clear(keyField);

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Select a Model")).toBeEnabled();
    });
  });
});
