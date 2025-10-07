import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, fireEvent } from "../../../tests/test-utils";
import LoggingSettings from "./LoggingSettings";

describe("LoggingSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("passes a number to updateCallbackVar when user inputs a number in NumericalInput", async () => {
    const user = userEvent.setup();
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
