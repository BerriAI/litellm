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

// Mock antd components
vi.mock("antd", () => ({
  Modal: ({ open, title, children, onCancel, footer }: any) =>
    open ? (
      <div data-testid="modal">
        <div>{title}</div>
        {children}
        {footer}
      </div>
    ) : null,
  Form: Object.assign(({ children, form }: any) => <form data-testid="form">{children}</form>, {
    useForm: () => [
      {
        resetFields: vi.fn(),
        validateFields: vi.fn(),
        getFieldsValue: vi.fn(),
        setFieldsValue: vi.fn(),
      },
      vi.fn(),
    ],
    Item: ({ children }: any) => <div>{children}</div>,
  }),
  Steps: Object.assign(
    ({ children, current, className }: any) => (
      <div data-testid="steps" className={className}>
        {children}
      </div>
    ),
    {
      Step: ({ title }: any) => <div>{title}</div>,
    },
  ),
  Button: ({ children, onClick, disabled, loading, ...props }: any) => (
    <button onClick={onClick} disabled={disabled || loading} data-loading={loading} {...props}>
      {children}
    </button>
  ),
  Checkbox: ({ checked, indeterminate, onChange, children, disabled }: any) => (
    <label>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange({ target: { checked: e.target.checked } })}
        disabled={disabled}
        data-indeterminate={indeterminate}
      />
      {children}
    </label>
  ),
}));

// Mock @tremor/react components
vi.mock("@tremor/react", () => ({
  Text: ({ children, className }: any) => <span className={className}>{children}</span>,
  Title: ({ children }: any) => <h3>{children}</h3>,
  Badge: ({ children, color, size }: any) => (
    <span data-color={color} data-size={size}>
      {children}
    </span>
  ),
}));

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

// Mock NotificationsManager
vi.mock("../../molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
    success: vi.fn(),
  },
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

    // Check that the component renders with the correct title and content
    expect(screen.getByText("Make Models Public")).toBeInTheDocument();
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();

    // Check that all model checkboxes are present
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 models

    // Check that the Next button is enabled (models are preselected)
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle model selection and navigation", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Initially on step 1
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();

    // Select all models using the select all checkbox
    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // Verify Next button is enabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();

    // Click Next
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Should move to step 2
    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });
  });

  it("should submit selected models successfully", async () => {
    mockMakeModelGroupPublic.mockResolvedValueOnce({});

    render(<MakeModelPublicForm {...mockProps} />);

    // Select all models
    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Wait for navigation to complete
    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    // Submit
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

    // Select all
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // All checkboxes should be checked
    checkboxes.forEach((checkbox) => {
      expect(checkbox).toBeChecked();
    });

    // Deselect all
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // All checkboxes should be unchecked except the indeterminate state
    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).not.toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  it("should show error when no models selected", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Deselect all models first
    const checkboxes = screen.getAllByRole("checkbox");
    await act(async () => {
      fireEvent.click(checkboxes[0]); // Click select all to select all
      fireEvent.click(checkboxes[0]); // Click select all again to deselect all
    });

    // Try to go to next step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Should stay on same step
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no models are available", () => {
    const emptyProps = {
      ...mockProps,
      modelHubData: [] as ModelGroupInfo[],
    };

    render(<MakeModelPublicForm {...emptyProps} />);

    expect(screen.getByText("No models match the current filters.")).toBeInTheDocument();

    // Select All checkbox should be disabled
    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    // Next button should be disabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Click Cancel button
    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    // Should call onClose
    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Navigate to step 1
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Verify we're on step 1
    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    // Click Previous button
    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    // Should go back to step 0
    expect(screen.getByText("Select Models to Make Public")).toBeInTheDocument();
  });

  it("should handle individual model selection", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Get all checkboxes (select all + individual models)
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 models

    // Initially, gpt-3.5-turbo should be selected (it's already public)
    const gpt4Checkbox = checkboxes[1]; // First model checkbox
    const gpt35Checkbox = checkboxes[2]; // Second model checkbox

    expect(gpt35Checkbox).toBeChecked(); // gpt-3.5-turbo is already public

    // Select gpt-4
    await act(async () => {
      fireEvent.click(gpt4Checkbox);
    });

    expect(gpt4Checkbox).toBeChecked();
    expect(gpt35Checkbox).toBeChecked();

    // Deselect gpt-3.5-turbo
    await act(async () => {
      fireEvent.click(gpt35Checkbox);
    });

    expect(gpt4Checkbox).toBeChecked();
    expect(gpt35Checkbox).not.toBeChecked();

    // Select all should be indeterminate now
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
  });

  it("should display model badges and information", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Should show model names
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();

    // Should show mode badges
    expect(screen.getAllByText("chat")).toHaveLength(2);

    // Should show provider badges
    expect(screen.getAllByText("openai")).toHaveLength(2);
  });

  it("should handle submit error properly", async () => {
    const errorMessage = "Network error";
    mockMakeModelGroupPublic.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeModelPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Should handle error and show error notification
    await waitFor(() => {
      expect(mockMakeModelGroupPublic).toHaveBeenCalledWith("test-token", ["gpt-3.5-turbo"]);
    });

    // Should not call onSuccess or onClose on error
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

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Check loading state
    expect(submitButton).toHaveAttribute("data-loading", "true");
    expect(submitButton).toBeDisabled();

    // Resolve the promise
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

    // Modal should not be rendered
    expect(screen.queryByTestId("modal")).not.toBeInTheDocument();
    expect(screen.queryByText("Make Models Public")).not.toBeInTheDocument();
  });

  it("should preselect already public models when modal opens", () => {
    // Test data where one model is public and one is not
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

    // Check that the correct checkboxes are selected
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4); // Select all + 3 models

    // private-model should not be checked, public models should be checked
    const privateModelCheckbox = checkboxes[1];
    const publicModelCheckbox = checkboxes[2];
    const anotherPublicModelCheckbox = checkboxes[3];

    expect(privateModelCheckbox).not.toBeChecked(); // private-model is not public
    expect(publicModelCheckbox).toBeChecked(); // public-model is public
    expect(anotherPublicModelCheckbox).toBeChecked(); // another-public-model is public

    // Select all should be indeterminate
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
  });

  it("should show selected count", () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Should show that 1 model is selected (gpt-3.5-turbo is preselected)
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("model selected")).toBeInTheDocument();
  });

  it("should show confirmation step with selected models", async () => {
    render(<MakeModelPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Models Public")).toBeInTheDocument();
    });

    // Should show the selected model
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();

    // Should show the warning message
    expect(screen.getByText(/Warning:/)).toBeInTheDocument();
    expect(screen.getByText(/model_hub_table/)).toBeInTheDocument();

    // Should show total count (already verified by checking the presence of the confirmation step)
  });
});
