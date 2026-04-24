import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import MakeModelPublicForm from "./MakeModelPublicForm";

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  [key: string]: any;
}

// Mock the networking function
vi.mock("../../networking", () => ({
  makeModelGroupPublic: vi.fn(),
}));

// Import the mocked function
import { makeModelGroupPublic } from "../../networking";
const mockMakeModelGroupPublic = vi.mocked(makeModelGroupPublic);

// Mock ModelFilters component
vi.mock("../../model_filters", () => ({
  default: ({ onFilteredDataChange, modelHubData }: any) => (
    <div data-testid="model-filters">
      <button data-testid="trigger-filter-change" onClick={() => onFilteredDataChange(modelHubData)}>
        Apply Filters
      </button>
    </div>
  ),
}));

describe("MakeModelPublicForm", () => {
  const mockProps = {
    visible: true,
    onClose: vi.fn(),
    accessToken: "test-token",
    modelHubData: [
      {
        model_group: "gpt-4",
        providers: ["openai"],
        max_input_tokens: 8192,
        max_output_tokens: 4096,
        input_cost_per_token: 0.03,
        output_cost_per_token: 0.06,
        mode: "chat",
        tpm: 10000,
        rpm: 200,
        supports_parallel_function_calling: true,
        supports_vision: false,
        supports_function_calling: true,
        supported_openai_params: ["temperature", "max_tokens"],
        is_public_model_group: false,
      },
      {
        model_group: "gpt-3.5-turbo",
        providers: ["openai"],
        max_input_tokens: 4096,
        max_output_tokens: 2048,
        input_cost_per_token: 0.0015,
        output_cost_per_token: 0.002,
        mode: "chat",
        tpm: 60000,
        rpm: 3500,
        supports_parallel_function_calling: false,
        supports_vision: false,
        supports_function_calling: true,
        supported_openai_params: ["temperature", "max_tokens"],
        is_public_model_group: true,
      },
    ] as ModelGroupInfo[],
    onSuccess: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should render the component", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    expect(screen.getByText("Make Models Public")).toBeInTheDocument();
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();
  });

  it("should initialize with correct state", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    expect(screen.getByText("Make Models Public")).toBeInTheDocument();
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle model selection and navigation", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();

    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();

    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });
  });

  it("should submit selected models successfully", async () => {
    mockMakeModelGroupPublic.mockResolvedValueOnce({});

    render(<MakeModelPublicForm {...mockProps} />);

    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeModelGroupPublic).toHaveBeenCalledWith("test-token", ["gpt-4", "gpt-3.5-turbo"]);
      expect(mockProps.onSuccess).toHaveBeenCalled();
      expect(mockProps.onClose).toHaveBeenCalled();
    });
  });

  it("should handle select all functionality", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    const selectAllCheckbox = checkboxes[0];

    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    checkboxes.forEach((checkbox) => {
      expect(checkbox).toBeChecked();
    });

    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).not.toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  it("should show error when no models selected", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    await act(async () => {
      fireEvent.click(checkboxes[0]);
    });
    await act(async () => {
      fireEvent.click(checkboxes[0]);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no models are available", () => {
    const emptyProps = {
      ...mockProps,
      modelHubData: [] as ModelGroupInfo[],
    };

    render(<MakeModelPublicForm {...emptyProps} />);

    expect(screen.getByText("No models match the current filters.")).toBeInTheDocument();

    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();
  });

  it("should handle individual model selection", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const gpt4Checkbox = checkboxes[1];
    const gpt35Checkbox = checkboxes[2];

    expect(gpt35Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(gpt4Checkbox);
    });

    expect(gpt4Checkbox).toBeChecked();
    expect(gpt35Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(gpt35Checkbox);
    });

    expect(gpt4Checkbox).toBeChecked();
    expect(gpt35Checkbox).not.toBeChecked();

    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
  });

  it("should display model badges and information", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();

    expect(screen.getAllByText("chat")).toHaveLength(2);
    expect(screen.getAllByText("openai")).toHaveLength(2);
  });

  it("should handle submit error properly", async () => {
    const errorMessage = "Network error";
    mockMakeModelGroupPublic.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeModelPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeModelGroupPublic).toHaveBeenCalledWith("test-token", ["gpt-3.5-turbo"]);
    });

    expect(mockProps.onSuccess).not.toHaveBeenCalled();
    expect(mockProps.onClose).not.toHaveBeenCalled();
  });

  it("should show loading state during submit", async () => {
    let resolvePromise: (value: any) => void = () => {};
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockMakeModelGroupPublic.mockReturnValueOnce(pendingPromise);

    render(<MakeModelPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Make Public" }));
    });

    const loadingButton = screen.getByRole("button", { name: "Making Public..." });
    expect(loadingButton).toHaveAttribute("data-loading", "true");
    expect(loadingButton).toBeDisabled();

    resolvePromise({});
    await waitFor(() => {
      expect(mockProps.onSuccess).toHaveBeenCalled();
      expect(mockProps.onClose).toHaveBeenCalled();
    });
  });

  it("should not render modal when visible is false", () => {
    const invisibleProps = {
      ...mockProps,
      visible: false,
    };

    render(<MakeModelPublicForm {...invisibleProps} />);

    expect(screen.queryByText("Make Models Public")).not.toBeInTheDocument();
  });

  it("should preselect already public models when modal opens", () => {
    const mixedPublicProps = {
      ...mockProps,
      modelHubData: [
        {
          model_group: "private-model",
          providers: ["openai"],
          is_public_model_group: false,
          mode: "chat",
        },
        {
          model_group: "public-model",
          providers: ["anthropic"],
          is_public_model_group: true,
          mode: "completion",
        },
        {
          model_group: "another-public-model",
          providers: ["cohere"],
          is_public_model_group: true,
          mode: "chat",
        },
      ] as ModelGroupInfo[],
    };

    render(<MakeModelPublicForm {...mixedPublicProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4);

    const privateModelCheckbox = checkboxes[1];
    const publicModelCheckbox = checkboxes[2];
    const anotherPublicModelCheckbox = checkboxes[3];

    expect(privateModelCheckbox).not.toBeChecked();
    expect(publicModelCheckbox).toBeChecked();
    expect(anotherPublicModelCheckbox).toBeChecked();

    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
  });

  it("should show selected count", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // "1 model selected" is rendered as `<strong>1</strong> model selected`
    const banner = screen.getByText((_content, node) => {
      if (!node) return false;
      if (node.tagName !== "P") return false;
      return (node.textContent ?? "").trim().replace(/\s+/g, " ") === "1 model selected";
    });
    expect(banner).toBeInTheDocument();
  });

  it("should show confirmation step with selected models", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();

    expect(screen.getByText(/Warning:/)).toBeInTheDocument();
    expect(screen.getByText(/model_hub_table/)).toBeInTheDocument();
  });
});
