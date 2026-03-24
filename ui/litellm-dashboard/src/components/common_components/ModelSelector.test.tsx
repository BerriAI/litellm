import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";
import ModelSelector from "./ModelSelector";
import { fetchAvailableModels } from "@/components/playground/llm_calls/fetch_models";

const mockModels = [
  { model_group: "gpt-4" },
  { model_group: "claude-3-opus" },
  { model_group: "gemini-pro" },
];

vi.mock("@/components/playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

describe("ModelSelector", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(fetchAvailableModels).mockResolvedValue(mockModels);
  });

  it("should render", () => {
    renderWithProviders(<ModelSelector accessToken="test-token" />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should show the label by default", () => {
    renderWithProviders(<ModelSelector accessToken="test-token" />);
    expect(screen.getByText("Select Model")).toBeInTheDocument();
  });

  it("should hide the label when showLabel is false", () => {
    renderWithProviders(
      <ModelSelector accessToken="test-token" showLabel={false} />
    );
    expect(screen.queryByText("Select Model")).not.toBeInTheDocument();
  });

  it("should show custom label text", () => {
    renderWithProviders(
      <ModelSelector accessToken="test-token" labelText="Pick a Model" />
    );
    expect(screen.getByText("Pick a Model")).toBeInTheDocument();
  });

  it("should show the custom model input when 'Enter custom model' is selected", async () => {
    const user = userEvent.setup();

    renderWithProviders(<ModelSelector accessToken="test-token" />);

    // Wait for models to load
    await waitFor(() => {
      expect(fetchAvailableModels).toHaveBeenCalledWith("test-token");
    });

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Enter custom model"));

    expect(
      screen.getByPlaceholderText("Enter custom model name")
    ).toBeInTheDocument();
  });

  it("should fetch models when accessToken is provided", async () => {
    renderWithProviders(<ModelSelector accessToken="test-token" />);

    await waitFor(() => {
      expect(fetchAvailableModels).toHaveBeenCalledWith("test-token");
    });
  });

  it("should not fetch models when accessToken is empty", () => {
    renderWithProviders(<ModelSelector accessToken="" />);

    expect(fetchAvailableModels).not.toHaveBeenCalled();
  });
});
