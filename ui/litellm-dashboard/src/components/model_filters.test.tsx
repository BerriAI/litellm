import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ModelFilters from "./model_filters";

const mockData = [
  {
    model_group: "gpt-4",
    providers: ["openai"],
    mode: "chat",
    supports_function_calling: true,
    supports_vision: false,
    supports_parallel_function_calling: false,
    is_public_model_group: true,
  },
  {
    model_group: "claude-3",
    providers: ["anthropic"],
    mode: "chat",
    supports_function_calling: true,
    supports_vision: true,
    supports_parallel_function_calling: false,
    is_public_model_group: true,
  },
  {
    model_group: "whisper-1",
    providers: ["openai"],
    mode: "audio_transcription",
    supports_function_calling: false,
    supports_vision: false,
    supports_parallel_function_calling: false,
    is_public_model_group: true,
  },
];

describe("ModelFilters", () => {
  const defaultProps = {
    modelHubData: mockData,
    onFilteredDataChange: vi.fn(),
  };

  it("should render", () => {
    render(<ModelFilters {...defaultProps} />);
    expect(screen.getByPlaceholderText(/search model names/i)).toBeInTheDocument();
  });

  it("should filter models by search term", async () => {
    const onFilteredDataChange = vi.fn();
    const user = userEvent.setup();
    render(<ModelFilters {...defaultProps} onFilteredDataChange={onFilteredDataChange} />);

    await user.type(screen.getByPlaceholderText(/search model names/i), "gpt");

    expect(onFilteredDataChange).toHaveBeenLastCalledWith(
      expect.arrayContaining([expect.objectContaining({ model_group: "gpt-4" })])
    );
  });

  it("should show provider dropdown with available providers", () => {
    render(<ModelFilters {...defaultProps} />);
    const providerSelect = screen.getByDisplayValue("All Providers");
    expect(providerSelect).toBeInTheDocument();
  });

  it("should show mode dropdown with available modes", () => {
    render(<ModelFilters {...defaultProps} />);
    expect(screen.getByDisplayValue("All Modes")).toBeInTheDocument();
  });

  it("should show features dropdown", () => {
    render(<ModelFilters {...defaultProps} />);
    expect(screen.getByDisplayValue("All Features")).toBeInTheDocument();
  });

  it("should show Clear Filters button when a filter is active", async () => {
    const user = userEvent.setup();
    render(<ModelFilters {...defaultProps} />);

    await user.type(screen.getByPlaceholderText(/search model names/i), "gpt");

    expect(screen.getByText("Clear Filters")).toBeInTheDocument();
  });

  it("should not show Clear Filters button when no filters are active", () => {
    render(<ModelFilters {...defaultProps} />);
    expect(screen.queryByText("Clear Filters")).not.toBeInTheDocument();
  });

  it("should render without Card wrapper when showFiltersCard is false", () => {
    const { container } = render(<ModelFilters {...defaultProps} showFiltersCard={false} />);
    // When showFiltersCard=false, it renders a plain div instead of a Card
    expect(container.querySelector(".tremor-Card-root")).toBeNull();
  });
});
