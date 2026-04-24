import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatUI from "./ChatUI";
import * as fetchModelsModule from "../llm_calls/fetch_models";

// Mock the fetchAvailableModels function
vi.mock("../llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

// Mock other networking functions that cause errors
vi.mock("../networking", () => ({
  tagListCall: vi.fn().mockResolvedValue({ data: [] }),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
}));

// Mock scrollIntoView which is not available in jsdom
beforeEach(() => {
  Element.prototype.scrollIntoView = () => { };
});

describe("ChatUI", () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
    sessionStorage.clear();

    // Mock scrollIntoView which is not available in JSDOM
    Element.prototype.scrollIntoView = vi.fn();

    // Mock the fetchAvailableModels to return test models
    (fetchModelsModule.fetchAvailableModels as any).mockResolvedValue([
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
    expect(getByText("Test Key")).toBeInTheDocument();
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

    // The endpoint selector is a shadcn Select (role=combobox). There is one
    // other combobox on the page (MCP servers) so we scope by the "Endpoint
    // Type" label.
    const endpointTypeText = getByText("Endpoint Type");
    const selectContainer = endpointTypeText.parentElement as HTMLElement;
    const endpointCombobox = selectContainer.querySelector('[role="combobox"]') as HTMLElement;
    expect(endpointCombobox).toBeInTheDocument();

    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(endpointCombobox);

    // Wait for the dropdown to appear and pick the audio_speech option
    const audioSpeechOption = await screen.findByRole("option", { name: "/v1/audio/speech" });
    await user.click(audioSpeechOption);

    // Verify the voice selector appears
    await waitFor(() => {
      expect(getByText("Voice")).toBeInTheDocument();
    });

    // The Voice select is still an antd Select component
    const voiceText = getByText("Voice");
    const voiceSelectContainer = voiceText.parentElement;
    const voiceSelectElement = voiceSelectContainer?.querySelector(".ant-select");
    expect(voiceSelectElement).toBeInTheDocument();
  });

  it("should allow the user to select a model", async () => {
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

    // Open the "Select Model" dropdown (AntD renders options in a portal)
    const selectModelLabel = getByText("Select Model");
    // The Select component is a sibling of the Text component, so we need to find it in the parent container
    const modelSelectContainer = selectModelLabel.closest("div");
    const modelSelect = modelSelectContainer?.querySelector(".ant-select-selector");
    expect(modelSelect).toBeTruthy();

    fireEvent.mouseDown(modelSelect!);

    await waitFor(() => {
      const model1Label = screen.getAllByText("Model 1");
      expect(model1Label.length).toBeGreaterThan(0);
    });
  });

  it("shows only chat-compatible models when chat endpoint is selected", async () => {
    (fetchModelsModule.fetchAvailableModels as any).mockResolvedValueOnce([
      { model_group: "ChatModel", mode: "chat" },
      { model_group: "SpeechModel", mode: "audio_speech" },
      { model_group: "ImageModel", mode: "image_generation" },
      { model_group: "ResponsesModel", mode: "responses" },
    ]);

    const { getByText, baseElement } = render(
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

    // Open endpoint selector (shadcn Select) and explicitly select /v1/chat/completions
    const endpointTypeText = getByText("Endpoint Type");
    const endpointCombobox = endpointTypeText.parentElement?.querySelector('[role="combobox"]') as HTMLElement;
    expect(endpointCombobox).toBeTruthy();
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(endpointCombobox);
    const chatOption = await screen.findByRole("option", { name: "/v1/chat/completions" });
    await user.click(chatOption);

    // Open model selector
    const selectModelLabel = getByText("Select Model");
    // The Select component is a sibling of the Text component, so we need to find it in the parent container
    const modelSelectContainer = selectModelLabel.closest("div");
    const modelSelect = modelSelectContainer?.querySelector(".ant-select-selector");
    expect(modelSelect).toBeTruthy();
    act(() => {
      fireEvent.mouseDown(modelSelect!);
    });

    await waitFor(() => {
      // Chat-compatible: ChatModel should be visible
      expect(screen.getAllByText("ChatModel").length).toBeGreaterThan(0);
      expect(screen.queryByText("SpeechModel")).toBeNull();
      expect(screen.queryByText("ImageModel")).toBeNull();
      expect(screen.queryByText("ResponsesModel")).toBeNull();
    });
  });

  /**
   * Tests that the 'Enter custom model' option is available in the model selector dropdown.
   * This ensures users can manually enter a model name if it's not in the list.
   */
  it("should show 'Enter custom model' option in model selector", async () => {
    const { getByText } = render(
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

    // Open the "Select Model" dropdown
    const selectModelLabel = getByText("Select Model");
    const modelSelectContainer = selectModelLabel.closest("div");
    const modelSelect = modelSelectContainer?.querySelector(".ant-select-selector");

    fireEvent.mouseDown(modelSelect!);

    await waitFor(() => {
      // Get all options in the dropdown (Ant Design renders these in a portal)
      const options = document.querySelectorAll(".ant-select-item-option-content");
      expect(options.length).toBeGreaterThan(0);
      // Check if the first option is 'Enter custom model'
      expect(options[0]).toHaveTextContent("Enter custom model");
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

    const endpointTypeText = screen.getByText("Endpoint Type");
    const endpointCombobox = endpointTypeText.parentElement?.querySelector('[role="combobox"]') as HTMLElement | null;
    expect(endpointCombobox).not.toBeNull();

    const user = (await import("@testing-library/user-event")).default.setup();
    const selectEndpointOption = async (label: string) => {
      await user.click(endpointCombobox!);
      const option = await screen.findByRole("option", { name: label });
      await user.click(option);
    };

    const getMcpSelect = () =>
      screen.getByText("MCP Servers").closest("div")?.querySelector(".ant-select") as HTMLElement | null;

    await selectEndpointOption("/v1/embeddings");

    const mcpSelect = getMcpSelect();
    expect(mcpSelect).not.toBeNull();

    await waitFor(() => {
      expect(mcpSelect).toHaveClass("ant-select-disabled");
    });

    await selectEndpointOption("/v1/chat/completions");

    await waitFor(() => {
      expect(mcpSelect).not.toHaveClass("ant-select-disabled");
    });
  });

  it("should show Simulate failure to test fallbacks in Model Settings when chat endpoint is selected", async () => {
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

    // Model Settings button only appears when a chat model is selected; select "Model 1" first
    const selectModelLabel = screen.getByText("Select Model");
    const modelSelectContainer = selectModelLabel.closest("div");
    const modelSelect = modelSelectContainer?.querySelector(".ant-select-selector");
    expect(modelSelect).toBeTruthy();

    await act(async () => {
      fireEvent.mouseDown(modelSelect!);
    });

    await waitFor(() => {
      expect(screen.getAllByText("Model 1").length).toBeGreaterThan(0);
    });

    // Ant Design Select options may not have role="option"; click the dropdown option by text
    const model1Options = screen.getAllByText("Model 1");
    await act(async () => {
      fireEvent.click(model1Options[model1Options.length - 1]);
    });

    await waitFor(() => {
      const modelSettingsButton = screen.getByTestId("model-settings-button");
      expect(modelSettingsButton).toBeInTheDocument();
    });

    const modelSettingsButton = screen.getByTestId("model-settings-button");
    await act(async () => {
      fireEvent.click(modelSettingsButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
      expect(screen.getByText(/Simulate failure to test fallbacks/i)).toBeInTheDocument();
    });

    const fallbacksCheckbox = screen.getByRole("checkbox", {
      name: /Simulate failure to test fallbacks/i,
    });
    expect(fallbacksCheckbox).not.toBeChecked();

    await act(async () => {
      fireEvent.click(fallbacksCheckbox);
    });

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: /Simulate failure to test fallbacks/i })).toBeChecked();
    });
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
      expect(screen.queryByText("Fill")).toBeNull();
    });

    const customProxyInput = screen.getByPlaceholderText("Optional: Enter custom proxy URL (e.g., http://localhost:5000)");
    expect(customProxyInput).toHaveValue(testProxyUrl);
  });

  it("should enable search functionality for MCP server selector", async () => {
    render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Test Key")).toBeInTheDocument();
    });

    const mcpServersText = screen.queryByText("MCP Servers");
    expect(mcpServersText).toBeInTheDocument();

    // The MCP Servers select is the antd Select whose placeholder is "Select
    // MCP servers". It lives below the "MCP Servers" label.
    const mcpSelectInput = document.querySelector<HTMLElement>(
      "input.ant-select-selection-search-input",
    );
    expect(mcpSelectInput).toBeInTheDocument();

    const mcpSelect = mcpSelectInput!.closest(".ant-select") as HTMLElement;
    const selector = mcpSelect.querySelector(".ant-select-selector") as HTMLElement;
    fireEvent.mouseDown(selector);

    await waitFor(() => {
      const allServersOption = screen.queryByText("All MCP Servers");
      if (allServersOption) {
        expect(allServersOption).toBeInTheDocument();
      }
    });

    // showSearch on antd Select yields a search input inside .ant-select
    expect(mcpSelect.querySelector("input.ant-select-selection-search-input")).toBeInTheDocument();
  });
});
