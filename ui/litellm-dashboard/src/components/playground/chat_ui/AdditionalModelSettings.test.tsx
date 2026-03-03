import { act, render, screen, waitFor } from "@testing-library/react";
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
      expect(sliders[0]).not.toBeDisabled();
    });

    const temperatureSlider = screen.getAllByRole("slider")[0];
    const maxTokensSlider = screen.getAllByRole("slider")[1];
    expect(temperatureSlider).not.toBeDisabled();
    expect(maxTokensSlider).not.toBeDisabled();
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
      <AdditionalModelSettings
        mockTestFallbacks={currentValue}
        onMockTestFallbacksChange={handleChange}
      />,
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

    rerender(
      <AdditionalModelSettings
        mockTestFallbacks={currentValue}
        onMockTestFallbacksChange={handleChange}
      />,
    );

    await act(async () => {
      await user.click(screen.getByRole("checkbox", { name: /Simulate failure to test fallbacks/i }));
    });

    await waitFor(() => {
      expect(onMockTestFallbacksChange).toHaveBeenCalledWith(false);
    });
  });
});
