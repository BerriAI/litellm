import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, fireEvent, within } from "../../../tests/test-utils";
import { CALLBACK_CONFIGS } from "../callback_info_helpers";
import LoggingSettings from "./LoggingSettings";

const STANDARD_CALLBACK_DYNAMIC_PARAMS_ACCEPTED_BY_PROXY = [
  "langfuse_public_key",
  "langfuse_secret",
  "langfuse_secret_key",
  "langfuse_host",
  "langfuse_prompt_version",
  "gcs_bucket_name",
  "gcs_path_service_account",
  "langsmith_api_key",
  "langsmith_project",
  "langsmith_base_url",
  "langsmith_sampling_rate",
  "langsmith_tenant_id",
  "humanloop_api_key",
  "arize_api_key",
  "arize_space_key",
  "arize_space_id",
  "posthog_api_key",
  "posthog_api_url",
  "wandb_api_key",
  "weave_project_id",
  "dd_api_key",
  "dd_site",
  "dd_agent_host",
  "dd_agent_port",
  "turn_off_message_logging",
  "litellm_disabled_callbacks",
];

const openIntegrationTypeDropdown = () => {
  const integrationTypeLabel = screen.getByText("Integration Type");
  const select = within(integrationTypeLabel.parentElement as HTMLElement).getByRole("combobox");
  fireEvent.mouseDown(select);
};

const integrationOption = (displayName: string): HTMLElement => {
  const options = Array.from(document.querySelectorAll(".ant-select-item-option"));
  const option = options.find((element) => element.textContent?.endsWith(displayName));
  if (!option) {
    throw new Error(`No integration option for ${displayName} in [${options.map((o) => o.textContent).join(", ")}]`);
  }
  return option as HTMLElement;
};

describe("LoggingSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("only declares callback vars the proxy accepts for key/team logging", () => {
    const declaredParams = CALLBACK_CONFIGS.flatMap((config) => Object.keys(config.dynamic_params));

    expect(declaredParams.length).toBeGreaterThan(0);
    expect(
      declaredParams.filter((param) => !STANDARD_CALLBACK_DYNAMIC_PARAMS_ACCEPTED_BY_PROXY.includes(param)),
    ).toEqual([]);
  });

  it("offers integrations whose credentials only come from the proxy config", () => {
    const initialValue = [{ callback_name: "", callback_type: "success", callback_vars: {} }];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);
    openIntegrationTypeDropdown();

    for (const displayName of ["Arize Phoenix", "Azure Blob Storage", "Datadog", "Datadog LLM Observability"]) {
      expect(integrationOption(displayName)).toBeInTheDocument();
    }
  });

  it("stores the internal callback name when a newly added integration is selected", () => {
    const mockOnChange = vi.fn();
    const initialValue = [{ callback_name: "", callback_type: "success", callback_vars: {} }];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={mockOnChange} />);
    openIntegrationTypeDropdown();
    fireEvent.click(integrationOption("Arize Phoenix"));

    expect(mockOnChange).toHaveBeenCalledWith([
      { callback_name: "arize_phoenix", callback_type: "success", callback_vars: {} },
    ]);
  });

  it("renders per-key credential inputs for an integration that supports them", () => {
    const initialValue = [{ callback_name: "gcs_bucket", callback_type: "success", callback_vars: {} }];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    expect(screen.getByPlaceholderText("os.environ/GCS_BUCKET_NAME")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("os.environ/GCS_PATH_SERVICE_ACCOUNT")).toBeInTheDocument();
  });

  it("explains where credentials come from for an integration without per-key credentials", () => {
    const initialValue = [{ callback_name: "arize_phoenix", callback_type: "success", callback_vars: {} }];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    expect(screen.queryByText("Integration Parameters")).toBeNull();
    expect(screen.getByText(/reads its credentials from the proxy environment\/config/)).toBeInTheDocument();
  });

  it("passes a number to updateCallbackVar when user inputs a number in NumericalInput", async () => {
    const mockOnChange = vi.fn();

    // Create initial config with a callback that has number parameters (LangSmith has langsmith_sampling_rate)
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={mockOnChange} />);

    // Find the numerical input for langsmith_sampling_rate
    const numericalInput = screen.getByPlaceholderText("os.environ/LANGSMITH_SAMPLING_RATE");
    expect(numericalInput).toBeInTheDocument();

    // Use fireEvent.change to directly set the value (more reliable for number inputs)
    fireEvent.change(numericalInput, { target: { value: "0.75" } });

    // Verify that onChange was called
    expect(mockOnChange).toHaveBeenCalled();

    // Get the last call to onChange
    const lastCall = mockOnChange.mock.calls[mockOnChange.mock.calls.length - 1];
    const updatedConfig = lastCall[0];

    // Verify the structure and that the value is stored as a string (as expected by the component)
    expect(updatedConfig).toHaveLength(1);
    expect(updatedConfig[0].callback_vars.langsmith_sampling_rate).toBe("0.75");
  });

  it("displays number type indicator and validation hint for number parameters", () => {
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    // Check for the "Number" badge
    expect(screen.getByText("Number")).toBeInTheDocument();

    // Check for the validation hint
    expect(screen.getByText("Value must be between 0 and 1")).toBeInTheDocument();

    // Check that the input has the correct step attribute
    const numericalInput = screen.getByPlaceholderText("os.environ/LANGSMITH_SAMPLING_RATE");
    expect(numericalInput).toHaveAttribute("step", "0.01");
  });

  it("handles number input and text input independently", async () => {
    const mockOnChange = vi.fn();

    // Start with some existing values to simulate a more realistic scenario
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {
          langsmith_sampling_rate: "0.3",
          langsmith_api_key: "initial-key",
        },
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={mockOnChange} />);

    // Find both number and text inputs
    const numericalInput = screen.getByPlaceholderText("os.environ/LANGSMITH_SAMPLING_RATE");
    const textInput = screen.getByPlaceholderText("os.environ/LANGSMITH_API_KEY");

    // Verify initial values are displayed
    expect(numericalInput).toHaveValue(0.3); // NumberInput shows numeric value
    expect(textInput).toHaveValue("initial-key");

    // Change the numerical input
    fireEvent.change(numericalInput, { target: { value: "0.5" } });

    // Verify numerical input change was recorded and preserves other values
    expect(mockOnChange).toHaveBeenCalled();
    let lastCall = mockOnChange.mock.calls[mockOnChange.mock.calls.length - 1];
    let updatedConfig = lastCall[0];
    expect(updatedConfig[0].callback_vars.langsmith_sampling_rate).toBe("0.5");
    expect(updatedConfig[0].callback_vars.langsmith_api_key).toBe("initial-key"); // Should preserve existing value

    // Change the text input (this tests that text inputs work independently)
    fireEvent.change(textInput, { target: { value: "test-api-key" } });

    // Verify text input change was also recorded
    lastCall = mockOnChange.mock.calls[mockOnChange.mock.calls.length - 1];
    updatedConfig = lastCall[0];
    expect(updatedConfig[0].callback_vars.langsmith_api_key).toBe("test-api-key");
    // The component preserves the original initial value since we're starting from initial state each time
    expect(updatedConfig[0].callback_vars.langsmith_sampling_rate).toBe("0.3"); // Preserves initial value
  });

  it("masks a sensitive parameter until the reveal toggle is used", async () => {
    const user = userEvent.setup({ delay: null });
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: { langsmith_api_key: "sk-secret-value" },
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    const apiKeyInput = screen.getByPlaceholderText("os.environ/LANGSMITH_API_KEY");
    expect(apiKeyInput).toHaveAttribute("type", "password");

    await user.click(screen.getByRole("button", { name: "Show password" }));
    expect(apiKeyInput).toHaveAttribute("type", "text");
    expect(apiKeyInput).toHaveValue("sk-secret-value");

    await user.click(screen.getByRole("button", { name: "Hide password" }));
    expect(apiKeyInput).toHaveAttribute("type", "password");
  });

  it("shows the bundled logo in the integration card header", () => {
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    expect(screen.getByAltText("LangSmith logo")).toHaveAttribute("src", "/_next/static/media/langsmith.png");
  });

  it("shows a letter avatar in the card header for a callback without a bundled logo", () => {
    const initialValue = [
      {
        callback_name: "custom_callback_api",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    expect(screen.getByText("Custom Callback API Configuration")).toBeInTheDocument();
    expect(screen.queryByAltText("Custom Callback API logo")).not.toBeInTheDocument();
    expect(screen.getByText("C")).toBeInTheDocument();
  });

  it("correctly handles numerical input with decimal values", () => {
    const mockOnChange = vi.fn();

    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={mockOnChange} />);

    const numericalInput = screen.getByPlaceholderText("os.environ/LANGSMITH_SAMPLING_RATE");

    // Test various decimal values
    const testValues = ["0.1", "0.25", "0.5", "0.75", "1.0"];

    testValues.forEach((value) => {
      fireEvent.change(numericalInput, { target: { value } });

      const lastCall = mockOnChange.mock.calls[mockOnChange.mock.calls.length - 1];
      const updatedConfig = lastCall[0];
      expect(updatedConfig[0].callback_vars.langsmith_sampling_rate).toBe(value);
    });
  });
});
