import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatUI from "./ChatUI";
import * as fetchModelsModule from "./llm_calls/fetch_models";

// Mock the fetchAvailableModels function
vi.mock("./llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

// Mock other networking functions that cause errors
vi.mock("../networking", () => ({
  tagListCall: vi.fn().mockResolvedValue({ data: [] }),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ data: [] }),
  mcpToolsCall: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
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
});
