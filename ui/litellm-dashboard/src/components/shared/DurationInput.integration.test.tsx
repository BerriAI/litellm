import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { DurationInput } from "@/components/shared/DurationInput";

function DurationForm({ base, initial }: { base: "minutes" | "hours"; initial: number }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <DurationInput label="Duration" base={base} value={value} onChange={setValue} max={10080} />
      <output aria-label="Saved duration">{value}</output>
    </>
  );
}

describe("Duration units", () => {
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
