import { render, screen, waitFor } from "@testing-library/react";
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
});
