import { render, waitFor, screen, fireEvent, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
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

describe("ChatUI", () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();

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
});
