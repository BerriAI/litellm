import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";
import { NumberInput } from "./NumberInput";

const DEFAULT_VALUE = 3;

const DefaultingHarness = () => {
  const [value, setValue] = useState(DEFAULT_VALUE);
  return <NumberInput value={value} onValueChange={(next) => setValue(next ?? DEFAULT_VALUE)} min={0} />;
};

describe("NumberInput", () => {
  it("should report the typed number", () => {
    const onValueChange = vi.fn();
    render(<NumberInput value={3} onValueChange={onValueChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "7" } });

    expect(onValueChange).toHaveBeenCalledWith(7);
  });

  it("should report null when the field is cleared", () => {
    const onValueChange = vi.fn();
    render(<NumberInput value={3} onValueChange={onValueChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "" } });

    expect(onValueChange).toHaveBeenCalledWith(null);
  });

  it("should stay empty after a clear that sends the parent back to its default", () => {
    render(<DefaultingHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue(null);
  });

  it("should accept a fresh number typed into the cleared field", () => {
    render(<DefaultingHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.change(input, { target: { value: "5" } });

    expect(input).toHaveValue(5);
  });

  it("should show the parent value again once the cleared field is blurred", () => {
    render(<DefaultingHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);

    expect(input).toHaveValue(DEFAULT_VALUE);
  });

  it("should follow the parent value after the draft is committed", () => {
    const { rerender } = render(<NumberInput value={3} onValueChange={vi.fn()} />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);
    rerender(<NumberInput value={9} onValueChange={vi.fn()} />);

    expect(input).toHaveValue(9);
  });

  it("should call a caller-supplied blur handler", () => {
    const onBlur = vi.fn();
    render(<NumberInput value={3} onValueChange={vi.fn()} onBlur={onBlur} />);

    fireEvent.blur(screen.getByRole("spinbutton"));

    expect(onBlur).toHaveBeenCalledTimes(1);
  });
});
