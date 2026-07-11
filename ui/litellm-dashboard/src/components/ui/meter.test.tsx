import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Meter, MeterIndicator, MeterLabel, MeterTrack } from "./meter";

const renderMeter = (value: number, max: number) =>
  render(
    <Meter value={value} max={max}>
      <MeterLabel>Seats</MeterLabel>
      <MeterTrack>
        <MeterIndicator />
      </MeterTrack>
    </Meter>,
  );

describe("Meter", () => {
  it("exposes the value and range through the meter role", () => {
    renderMeter(5, 10);
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "5");
    expect(meter).toHaveAttribute("aria-valuemax", "10");
    expect(screen.getByText("Seats")).toHaveAttribute("data-slot", "meter-label");
  });

  it("fills the indicator proportionally to value/max", () => {
    const { container } = renderMeter(5, 10);
    const indicator = container.querySelector('[data-slot="meter-indicator"]');
    expect(indicator).toHaveStyle({ width: "50%" });
  });

  it("caps the indicator at 100% when the value exceeds the max", () => {
    const { container } = renderMeter(15, 10);
    const indicator = container.querySelector('[data-slot="meter-indicator"]');
    expect(indicator).toHaveStyle({ width: "100%" });
  });
});
