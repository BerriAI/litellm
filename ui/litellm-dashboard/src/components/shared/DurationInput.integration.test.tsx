import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { DurationInput } from "@/components/shared/DurationInput";

function DurationForm({ base, initial, max }: { base: "minutes" | "hours"; initial: number; max?: number }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <DurationInput label="Duration" base={base} value={value} onChange={setValue} max={max} />
      <output aria-label="Saved duration">{value}</output>
    </>
  );
}

describe("Duration units", () => {
  it.each([undefined, 10080])("validates large durations against an optional maximum of %s", (max) => {
    render(<DurationForm base="hours" initial={1} max={max} />);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Duration" }), { target: { value: 240000 } });
    expect(screen.getByLabelText("Saved duration")).toHaveTextContent("240000");
    if (max === undefined) {
      expect(screen.getByRole("spinbutton", { name: "Duration" })).toBeValid();
    } else {
      expect(screen.getByRole("spinbutton", { name: "Duration" })).toBeInvalid();
    }
  });
  it.each([
    { base: "hours" as const, initial: 24, unit: "1", displayed: 24 },
    { base: "minutes" as const, initial: 60, unit: "1", displayed: 60 },
  ])("preserves $initial $base when changing its display unit", ({ base, initial, unit, displayed }) => {
    render(<DurationForm base={base} initial={initial} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Duration unit" }), { target: { value: unit } });
    expect(screen.getByRole("spinbutton", { name: "Duration" })).toHaveValue(displayed);
    expect(screen.getByLabelText("Saved duration")).toHaveTextContent(String(initial));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Duration" }), { target: { value: 7 } });
    expect(screen.getByLabelText("Saved duration")).toHaveTextContent("7");
  });
});
