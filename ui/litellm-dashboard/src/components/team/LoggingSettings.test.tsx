import { readFileSync } from "fs";
import { resolve } from "path";
import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, fireEvent } from "../../../tests/test-utils";
import LoggingSettings from "./LoggingSettings";

const SOURCE_PATH = resolve(process.cwd(), "src/components/team/LoggingSettings.tsx");

const HARDCODED_PALETTE =
  /\b(?:text|bg|border|hover:bg|hover:text|hover:border|dark:bg|dark:text|dark:border|ring|divide|fill|stroke)-(?:gray|slate|zinc|neutral|stone|red|blue|green|yellow|amber|orange|indigo|purple|pink|rose|teal|cyan|sky|violet|fuchsia|lime|emerald)-\d+(?:\/\d+)?\b/g;

const SEMANTIC_TOKEN =
  /\b(?:text|bg|border|hover:bg|hover:text|ring|divide|fill|stroke)-(?:foreground|muted-foreground|muted|background|card|popover|primary|secondary|destructive|border|input|accent|ring)(?:-foreground)?(?:\/\d+)?\b/g;

describe("LoggingSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("styles itself from semantic tokens instead of hardcoded palette classes", () => {
    const source = readFileSync(SOURCE_PATH, "utf8");

    expect(source).toContain("const LoggingSettings");
    expect(source.match(SEMANTIC_TOKEN) ?? []).not.toHaveLength(0);
    expect(source.match(HARDCODED_PALETTE) ?? []).toHaveLength(0);
  });

  it("keeps the remove button destructive on hover instead of the ghost variant's foreground", () => {
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={vi.fn()} />);

    const remove = screen.getByRole("button", { name: "Remove" });
    expect(remove).toHaveClass("hover:text-destructive/80");
    expect(remove).not.toHaveClass("hover:text-foreground");
  });

  it("reports the chosen event type when a different option is picked", async () => {
    const user = userEvent.setup({ delay: null });
    const mockOnChange = vi.fn();
    const initialValue = [
      {
        callback_name: "langsmith",
        callback_type: "success",
        callback_vars: {},
      },
    ];

    renderWithProviders(<LoggingSettings value={initialValue} onChange={mockOnChange} />);

    await user.click(screen.getByRole("combobox", { name: "Event Type" }));
    await user.click(await screen.findByRole("option", { name: "Failure Only" }));

    expect(mockOnChange).toHaveBeenCalledWith([expect.objectContaining({ callback_type: "failure" })]);
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
