import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import AdditionalModelSettings from "./AdditionalModelSettings";

describe("AdditionalModelSettings", () => {
  it("should render correctly", () => {
    const { container } = render(<AdditionalModelSettings />);
    expect(container).toBeTruthy();

    expect(screen.getByText("Use Advanced Parameters")).toBeInTheDocument();

    expect(screen.getByText("Temperature")).toBeInTheDocument();
    expect(screen.getByText("Max Tokens")).toBeInTheDocument();
  });

  it("should enable the sliders when the checkbox is checked", async () => {
    const user = userEvent.setup();
    const mockOnTemperatureChange = vi.fn();
    const mockOnMaxTokensChange = vi.fn();

    render(
      <AdditionalModelSettings
        onTemperatureChange={mockOnTemperatureChange}
        onMaxTokensChange={mockOnMaxTokensChange}
      />,
    );

    const checkbox = screen.getByRole("checkbox", { name: /Use Advanced Parameters/i });
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).not.toBeChecked();

    await user.click(checkbox);

    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: /Use Advanced Parameters/i })).toBeChecked();
    });

    await waitFor(() => {
      const sliders = screen.getAllByRole("slider");
      expect(sliders.length).toBeGreaterThan(0);
      expect(sliders[0]).toBeEnabled();
    });

    const temperatureSlider = screen.getAllByRole("slider")[0];
    const maxTokensSlider = screen.getAllByRole("slider")[1];
    expect(temperatureSlider).toBeEnabled();
    expect(maxTokensSlider).toBeEnabled();
  });

  it("should not show Stream responses when onStreamingChange is not provided", () => {
    render(<AdditionalModelSettings />);
    expect(screen.queryByText(/Stream responses/i)).not.toBeInTheDocument();
  });

  it("should render Stream responses checked by default and report unchecking it", async () => {
    const user = userEvent.setup();
    const onStreamingChange = vi.fn();

    render(<AdditionalModelSettings onStreamingChange={onStreamingChange} />);

    const streamingCheckbox = screen.getByRole("checkbox", { name: /Stream responses/i });
    expect(streamingCheckbox).toBeChecked();

    await act(async () => {
      await user.click(streamingCheckbox);
    });

    await waitFor(() => {
      expect(onStreamingChange).toHaveBeenCalledWith(false);
    });
  });

  it("should keep the streaming toggle but drop advanced params when showAdvancedParams is false", () => {
    render(<AdditionalModelSettings showAdvancedParams={false} onStreamingChange={vi.fn()} />);

    expect(screen.getByRole("checkbox", { name: /Stream responses/i })).toBeInTheDocument();
    expect(screen.queryByText("Use Advanced Parameters")).not.toBeInTheDocument();
    expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
    expect(screen.queryByText("Max Tokens")).not.toBeInTheDocument();
  });

  it("should reflect a disabled streaming setting from props", () => {
    render(<AdditionalModelSettings streamingEnabled={false} onStreamingChange={vi.fn()} />);

    expect(screen.getByRole("checkbox", { name: /Stream responses/i })).not.toBeChecked();
  });

  it("should not show Simulate failure to test fallbacks when onMockTestFallbacksChange is not provided", () => {
    render(<AdditionalModelSettings />);
    expect(screen.queryByText(/Simulate failure to test fallbacks/i)).not.toBeInTheDocument();
  });

  it("should show and toggle Simulate failure to test fallbacks when callback is provided", async () => {
    const user = userEvent.setup();
    const onMockTestFallbacksChange = vi.fn();
    let currentValue = false;
    const handleChange = (value: boolean) => {
      currentValue = value;
      onMockTestFallbacksChange(value);
    };

    const { rerender } = render(
      <AdditionalModelSettings mockTestFallbacks={currentValue} onMockTestFallbacksChange={handleChange} />,
    );

    const fallbacksCheckbox = screen.getByRole("checkbox", {
      name: /Simulate failure to test fallbacks/i,
    });
    expect(fallbacksCheckbox).toBeInTheDocument();
    expect(fallbacksCheckbox).not.toBeChecked();

    await act(async () => {
      await user.click(fallbacksCheckbox);
    });

    await waitFor(() => {
      expect(onMockTestFallbacksChange).toHaveBeenCalledWith(true);
    });

    rerender(<AdditionalModelSettings mockTestFallbacks={currentValue} onMockTestFallbacksChange={handleChange} />);

    await act(async () => {
      await user.click(screen.getByRole("checkbox", { name: /Simulate failure to test fallbacks/i }));
    });

    await waitFor(() => {
      expect(onMockTestFallbacksChange).toHaveBeenCalledWith(false);
    });
  });

  it("should keep a half-typed decimal temperature instead of rewriting it", async () => {
    const onTemperatureChange = vi.fn();

    render(<AdditionalModelSettings useAdvancedParams onTemperatureChange={onTemperatureChange} />);

    const temperatureField = screen.getByLabelText("Temperature value") as HTMLInputElement;
    fireEvent.change(temperatureField, { target: { value: "0." } });

    expect(temperatureField.value).toBe("0.");

    fireEvent.change(temperatureField, { target: { value: "0.5" } });

    expect(temperatureField.value).toBe("0.5");
    expect(onTemperatureChange).toHaveBeenLastCalledWith(0.5);
  });

  it("should let the max tokens field be cleared instead of snapping to a value", async () => {
    const user = userEvent.setup();
    const onMaxTokensChange = vi.fn();

    render(<AdditionalModelSettings useAdvancedParams onMaxTokensChange={onMaxTokensChange} />);

    const maxTokensField = screen.getByLabelText("Max tokens value");
    await user.clear(maxTokensField);

    expect((maxTokensField as HTMLInputElement).value).toBe("");
  });

  it("should clamp an out-of-range temperature once the field is left", async () => {
    const user = userEvent.setup();
    const onTemperatureChange = vi.fn();

    render(<AdditionalModelSettings useAdvancedParams onTemperatureChange={onTemperatureChange} />);

    const temperatureField = screen.getByLabelText("Temperature value");
    await user.clear(temperatureField);
    fireEvent.change(temperatureField, { target: { value: "9" } });
    await user.tab();

    expect((temperatureField as HTMLInputElement).value).toBe("2");
    expect(onTemperatureChange).toHaveBeenLastCalledWith(2);
  });
});
