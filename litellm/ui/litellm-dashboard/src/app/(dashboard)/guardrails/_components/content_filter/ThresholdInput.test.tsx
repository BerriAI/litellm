import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ThresholdInput } from "./ThresholdInput";

const Harness = ({
  initial = 0.7,
  onValueChange,
}: {
  initial?: number;
  onValueChange?: (v: number | null) => void;
}) => {
  const [value, setValue] = useState(initial);
  return (
    <ThresholdInput
      value={value}
      min={0}
      max={1}
      step={0.05}
      onValueChange={(next) => {
        onValueChange?.(next);
        setValue(next ?? initial);
      }}
    />
  );
};

describe("ThresholdInput", () => {
  it("displays the value at the precision of the step", () => {
    render(<Harness initial={0.7} />);

    expect(screen.getByRole("spinbutton")).toHaveValue("0.70");
  });

  it("reports the parsed value while typing and null once the field is empty", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness onValueChange={onValueChange} />);

    const input = screen.getByRole("spinbutton");
    await user.clear(input);
    expect(onValueChange).toHaveBeenLastCalledWith(null);

    fireEvent.change(input, { target: { value: "0.55" } });
    expect(onValueChange).toHaveBeenLastCalledWith(0.55);
  });

  it("clamps a value above the maximum when the field loses focus", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness onValueChange={onValueChange} />);

    const input = screen.getByRole("spinbutton");
    await user.clear(input);
    fireEvent.change(input, { target: { value: "5" } });
    await user.tab();

    expect(onValueChange).toHaveBeenLastCalledWith(1);
    expect(input).toHaveValue("1.00");
  });

  it("steps by the step on the arrow keys and stops at the bounds", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness initial={0.95} onValueChange={onValueChange} />);

    const input = screen.getByRole("spinbutton");
    await user.click(input);
    await user.keyboard("{ArrowUp}");
    expect(onValueChange).toHaveBeenLastCalledWith(1);

    await user.keyboard("{ArrowUp}");
    expect(onValueChange).toHaveBeenLastCalledWith(1);

    await user.keyboard("{ArrowDown}");
    expect(onValueChange).toHaveBeenLastCalledWith(0.95);
  });

  it("exposes the bounds and the current value to assistive technology", () => {
    render(<Harness initial={0.45} />);

    const input = screen.getByRole("spinbutton");
    expect(input).toHaveAttribute("aria-valuemin", "0");
    expect(input).toHaveAttribute("aria-valuemax", "1");
    expect(input).toHaveAttribute("aria-valuenow", "0.45");
  });
});
