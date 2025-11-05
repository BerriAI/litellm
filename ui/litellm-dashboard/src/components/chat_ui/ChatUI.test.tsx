import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatUI from "./ChatUI";
import * as fetchModelsModule from "./llm_calls/fetch_models";
import * as chatCompletionModule from "./llm_calls/chat_completion";

// Mock the fetchAvailableModels function
vi.mock("./llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

// Mock the chat completion request
vi.mock("./llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn(),
}));

// Mock other networking functions that cause errors
vi.mock("../networking", () => ({
  tagListCall: vi.fn().mockResolvedValue({ data: [] }),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ data: [] }),
  mcpToolsCall: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
}));

// Mock scrollIntoView which is not available in jsdom
beforeEach(() => {
  Element.prototype.scrollIntoView = () => {};
});

describe("ChatUI", () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
    sessionStorage.clear();

    // Mock scrollIntoView which is not available in JSDOM
    Element.prototype.scrollIntoView = vi.fn();

    // Mock the fetchAvailableModels to return test models
    vi.mocked(fetchModelsModule.fetchAvailableModels).mockResolvedValue([
      { model_group: "Model 1", mode: "chat" },
      { model_group: "Model 2", mode: "chat" },
      { model_group: "Model 3", mode: "chat" },
    ]);
  });

  it("should render the chat UI", async () => {
    const { getByText } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });
  });

  it("should render the chat UI with selected models", async () => {
    const { getByText, container, getAllByText } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    // Wait for the first model to be auto-selected (default behavior)
    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    // Now click the Select dropdown to open it and check other models
    const selectComponent = container.querySelectorAll(".ant-select-selector")[1]; // Second select is for models
    expect(selectComponent).toBeTruthy();

    // Click to open the dropdown
    fireEvent.mouseDown(selectComponent!);

    // Wait for dropdown to open and check for all models
    await waitFor(() => {
      const dropdown = document.querySelector(".ant-select-dropdown");
      expect(dropdown).toBeTruthy();

      // All three models should be in the dropdown as options
      // Use getAllByTitle since Ant Design Select renders options with title attributes
      const model2Elements = screen.queryAllByText("Model 2");
      const model3Elements = screen.queryAllByText("Model 3");

      expect(model2Elements.length).toBeGreaterThan(0);
      expect(model3Elements.length).toBeGreaterThan(0);
    });
  });

  it("should allow users to pick multiple models and render badges for selected models", async () => {
    const { getByText, container } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    const selectComponent = container.querySelectorAll(".ant-select-selector")[1];
    expect(selectComponent).toBeTruthy();

    fireEvent.mouseDown(selectComponent!);

    let model2Option: HTMLElement | null = null;
    await waitFor(() => {
      const dropdown = document.querySelector(".ant-select-dropdown");
      expect(dropdown).toBeTruthy();
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model2Option = Array.from(options).find((opt) => opt.textContent === "Model 2") as HTMLElement;
      expect(model2Option).toBeTruthy();
    });

    fireEvent.click(model2Option!);

    await waitFor(() => {
      const model1Badges = screen.getAllByText("Model 1");
      const model2Badges = screen.getAllByText("Model 2");

      expect(model1Badges.length).toBeGreaterThan(0);
      expect(model2Badges.length).toBeGreaterThan(0);
    });

    fireEvent.mouseDown(selectComponent!);

    let model3Option: HTMLElement | null = null;
    await waitFor(() => {
      const dropdown = document.querySelector(".ant-select-dropdown");
      expect(dropdown).toBeTruthy();
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model3Option = Array.from(options).find((opt) => opt.textContent === "Model 3") as HTMLElement;
      expect(model3Option).toBeTruthy();
    });

    fireEvent.click(model3Option!);

    // Verify all three models have badges rendered
    await waitFor(() => {
      const model1Badges = screen.getAllByText("Model 1");
      const model2Badges = screen.getAllByText("Model 2");
      const model3Badges = screen.getAllByText("Model 3");

      expect(model1Badges.length).toBeGreaterThan(0);
      expect(model2Badges.length).toBeGreaterThan(0);
      expect(model3Badges.length).toBeGreaterThan(0);
    });
    const tags = container.querySelectorAll(".ant-tag");
    expect(tags.length).toBe(3);
  });

  it("should manage selectedModels state and its side effects", async () => {
    const { getByText, container } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    await waitFor(() => {
      const storedModels = sessionStorage.getItem("selectedModels");
      expect(storedModels).toBeTruthy();
      const parsedModels = JSON.parse(storedModels!);
      expect(parsedModels).toContain("Model 1");
    });

    const selectComponent = container.querySelectorAll(".ant-select-selector")[1];

    fireEvent.mouseDown(selectComponent!);

    let model2Option: HTMLElement | null = null;
    await waitFor(() => {
      const dropdown = document.querySelector(".ant-select-dropdown");
      expect(dropdown).toBeTruthy();
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model2Option = Array.from(options).find((opt) => opt.textContent === "Model 2") as HTMLElement;
      expect(model2Option).toBeTruthy();
    });

    fireEvent.click(model2Option!);

    await waitFor(() => {
      const model2Badges = screen.getAllByText("Model 2");
      expect(model2Badges.length).toBeGreaterThan(0);
    });

    await waitFor(() => {
      const storedModels = sessionStorage.getItem("selectedModels");
      const parsedModels = JSON.parse(storedModels!);
      expect(parsedModels).toContain("Model 1");
      expect(parsedModels).toContain("Model 2");
      expect(parsedModels.length).toBe(2);
    });

    const tags = container.querySelectorAll(".ant-tag");
    expect(tags.length).toBe(2);

    const firstTagCloseButton = tags[0].querySelector(".ant-tag-close-icon");
    expect(firstTagCloseButton).toBeTruthy();
    fireEvent.click(firstTagCloseButton!);

    await waitFor(() => {
      const storedModels = sessionStorage.getItem("selectedModels");
      const parsedModels = JSON.parse(storedModels!);
      expect(parsedModels).not.toContain("Model 1");
      expect(parsedModels).toContain("Model 2");
      expect(parsedModels.length).toBe(1);
    });

    await waitFor(() => {
      const remainingTags = container.querySelectorAll(".ant-tag");
      expect(remainingTags.length).toBe(1);
    });
  });

  it("should show the voice selector when the endpoint type is audio_speech", async () => {
    const { getByText, container } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    // Wait for the component to render
    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    // Find the endpoint selector by looking for the "Endpoint Type:" text and its associated Select
    const endpointTypeText = getByText("Endpoint Type:");
    const selectContainer = endpointTypeText.parentElement;
    const selectElement = selectContainer?.querySelector(".ant-select-selector");

    expect(selectElement).toBeInTheDocument();

    // Click on the select to open the dropdown
    if (selectElement) {
      fireEvent.mouseDown(selectElement);
    }

    // Wait for the dropdown to appear and find the audio_speech option
    await waitFor(() => {
      const audioSpeechOption = screen.getByText("/v1/audio/speech");
      expect(audioSpeechOption).toBeInTheDocument();
    });

    // Click on the audio_speech option
    const audioSpeechOption = screen.getByText("/v1/audio/speech");
    fireEvent.click(audioSpeechOption);

    // Verify the voice selector appears
    await waitFor(() => {
      expect(getByText("Voice")).toBeInTheDocument();
    });

    // Verify the voice select component is present
    const voiceText = getByText("Voice");
    const voiceSelectContainer = voiceText.parentElement;
    const voiceSelectElement = voiceSelectContainer?.querySelector(".ant-select");
    expect(voiceSelectElement).toBeInTheDocument();
  });

  it("should send requests to all selected models when multiple models are selected", async () => {
    // Mock the chat completion function to simulate successful requests
    const mockChatCompletion = vi.mocked(chatCompletionModule.makeOpenAIChatCompletionRequest);
    mockChatCompletion.mockImplementation(async (chatHistory, updateUI, selectedModel) => {
      // Simulate streaming response
      updateUI("Hello from ", selectedModel);
      updateUI(selectedModel, selectedModel);
    });

    const { getByText, container } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    // Wait for Model 1 to be auto-selected
    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    // Open model selector and select Model 2
    const selectComponent = container.querySelectorAll(".ant-select-selector")[1];
    fireEvent.mouseDown(selectComponent!);

    let model2Option: HTMLElement | null = null;
    await waitFor(() => {
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model2Option = Array.from(options).find((opt) => opt.textContent === "Model 2") as HTMLElement;
      expect(model2Option).toBeTruthy();
    });

    fireEvent.click(model2Option!);

    // Wait for both models to be selected
    await waitFor(() => {
      const tags = container.querySelectorAll(".ant-tag");
      expect(tags.length).toBe(2);
    });

    // Find the text area and type a message
    const textArea = container.querySelector("textarea");
    expect(textArea).toBeTruthy();

    fireEvent.change(textArea!, { target: { value: "Hello, models!" } });

    // Find and click the send button
    const sendButton = container.querySelector('button[class*="bg-blue-600"]');
    expect(sendButton).toBeTruthy();
    fireEvent.click(sendButton!);

    // Verify that makeOpenAIChatCompletionRequest was called twice (once for each model)
    await waitFor(
      () => {
        expect(mockChatCompletion).toHaveBeenCalledTimes(2);
      },
      { timeout: 3000 },
    );

    // Get all calls to the mock function
    const calls = mockChatCompletion.mock.calls;

    // Verify both models were called
    const modelsCalled = calls.map((call) => call[2]); // 3rd argument is the model
    expect(modelsCalled).toContain("Model 1");
    expect(modelsCalled).toContain("Model 2");

    // Verify that both calls received the user message
    calls.forEach((call) => {
      const chatHistory = call[0]; // First argument is chat history
      const lastMessage = chatHistory[chatHistory.length - 1];
      expect(lastMessage.role).toBe("user");
      expect(lastMessage.content).toBe("Hello, models!");
    });
  });

  it("should render separate chat boxes for each model response", async () => {
    // Mock the chat completion function to simulate responses from different models
    const mockChatCompletion = vi.mocked(chatCompletionModule.makeOpenAIChatCompletionRequest);
    mockChatCompletion.mockImplementation(async (chatHistory, updateUI, selectedModel) => {
      // Simulate different responses from different models
      updateUI(`Response from ${selectedModel}`, selectedModel);
    });

    const { getByText, container, getAllByText } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    // Wait for Model 1 to be auto-selected
    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    // Select Model 2
    const selectComponent = container.querySelectorAll(".ant-select-selector")[1];
    fireEvent.mouseDown(selectComponent!);

    let model2Option: HTMLElement | null = null;
    await waitFor(() => {
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model2Option = Array.from(options).find((opt) => opt.textContent === "Model 2") as HTMLElement;
      expect(model2Option).toBeTruthy();
    });

    fireEvent.click(model2Option!);

    // Select Model 3
    fireEvent.mouseDown(selectComponent!);

    let model3Option: HTMLElement | null = null;
    await waitFor(() => {
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model3Option = Array.from(options).find((opt) => opt.textContent === "Model 3") as HTMLElement;
      expect(model3Option).toBeTruthy();
    });

    fireEvent.click(model3Option!);

    // Wait for all three models to be selected
    await waitFor(() => {
      const tags = container.querySelectorAll(".ant-tag");
      expect(tags.length).toBe(3);
    });

    // Send a message
    const textArea = container.querySelector("textarea");
    fireEvent.change(textArea!, { target: { value: "Hello to all!" } });

    const sendButton = container.querySelector('button[class*="bg-blue-600"]');
    fireEvent.click(sendButton!);

    // Wait for all responses to be rendered
    await waitFor(
      () => {
        // Check that we have responses from each model
        expect(screen.getByText("Response from Model 1")).toBeInTheDocument();
        expect(screen.getByText("Response from Model 2")).toBeInTheDocument();
        expect(screen.getByText("Response from Model 3")).toBeInTheDocument();
      },
      { timeout: 3000 },
    );

    // Verify that each model's response is in a separate chat box with the model badge
    const chatMessages = container.querySelectorAll('[class*="inline-block max-w-"]');
    // Should have 4 messages total: 1 user message + 3 assistant messages (one from each model)
    expect(chatMessages.length).toBeGreaterThanOrEqual(4);

    // Verify each model badge appears in the assistant messages
    const modelBadges = container.querySelectorAll(".bg-gray-100.text-gray-600");
    const badgeTexts = Array.from(modelBadges).map((badge) => badge.textContent);
    expect(badgeTexts).toContain("Model 1");
    expect(badgeTexts).toContain("Model 2");
    expect(badgeTexts).toContain("Model 3");
  });

  it("should only send respective model messages in follow-up conversations", async () => {
    // Track all calls to the chat completion function
    const mockChatCompletion = vi.mocked(chatCompletionModule.makeOpenAIChatCompletionRequest);
    const callHistory: any[] = [];

    mockChatCompletion.mockImplementation(async (chatHistory, updateUI, selectedModel) => {
      // Store the chat history for each call
      callHistory.push({
        model: selectedModel,
        messages: [...chatHistory],
      });
      // Simulate response
      updateUI(`Response from ${selectedModel}`, selectedModel);
    });

    const { getByText, container } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Test Key")).toBeInTheDocument();
    });

    // Wait for Model 1 to be auto-selected
    await waitFor(() => {
      expect(getByText("Model 1")).toBeInTheDocument();
    });

    // Select Model 2
    const selectComponent = container.querySelectorAll(".ant-select-selector")[1];
    fireEvent.mouseDown(selectComponent!);

    let model2Option: HTMLElement | null = null;
    await waitFor(() => {
      const options = document.querySelectorAll(".ant-select-item-option-content");
      model2Option = Array.from(options).find((opt) => opt.textContent === "Model 2") as HTMLElement;
      expect(model2Option).toBeTruthy();
    });

    fireEvent.click(model2Option!);

    // Wait for both models to be selected
    await waitFor(() => {
      const tags = container.querySelectorAll(".ant-tag");
      expect(tags.length).toBe(2);
    });

    // Send first message
    const textArea = container.querySelector("textarea");
    fireEvent.change(textArea!, { target: { value: "First message" } });

    const sendButton = container.querySelector('button[class*="bg-blue-600"]');
    fireEvent.click(sendButton!);

    // Wait for first responses
    await waitFor(
      () => {
        expect(mockChatCompletion).toHaveBeenCalledTimes(2);
      },
      { timeout: 3000 },
    );

    // Clear call history for the second round
    callHistory.length = 0;
    mockChatCompletion.mockClear();

    // Send second message
    fireEvent.change(textArea!, { target: { value: "Second message" } });
    fireEvent.click(sendButton!);

    // Wait for second responses
    await waitFor(
      () => {
        expect(mockChatCompletion).toHaveBeenCalledTimes(2);
      },
      { timeout: 3000 },
    );

    // Verify that each model only received its own assistant messages in the history
    const model1Call = callHistory.find((call) => call.model === "Model 1");
    const model2Call = callHistory.find((call) => call.model === "Model 2");

    expect(model1Call).toBeTruthy();
    expect(model2Call).toBeTruthy();

    // Model 1's chat history should contain:
    // - First user message
    // - Model 1's assistant response
    // - Second user message
    const model1AssistantMessages = model1Call.messages.filter((msg: any) => msg.role === "assistant");
    model1AssistantMessages.forEach((msg: any) => {
      // Should not contain Model 2's responses (content shouldn't be "Response from Model 2")
      expect(msg.content).not.toContain("Response from Model 2");
    });

    // Model 2's chat history should contain:
    // - First user message
    // - Model 2's assistant response
    // - Second user message
    const model2AssistantMessages = model2Call.messages.filter((msg: any) => msg.role === "assistant");
    model2AssistantMessages.forEach((msg: any) => {
      // Should not contain Model 1's responses
      expect(msg.content).not.toContain("Response from Model 1");
    });

    // Verify user messages are included in both
    const model1UserMessages = model1Call.messages.filter((msg: any) => msg.role === "user");
    const model2UserMessages = model2Call.messages.filter((msg: any) => msg.role === "user");

    expect(model1UserMessages.length).toBe(2); // Both user messages
    expect(model2UserMessages.length).toBe(2); // Both user messages
  });
});
