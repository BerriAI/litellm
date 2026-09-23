import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";
import { NumberInput } from "./NumberInput";

const STORED_VALUE = 3;

const StatefulHarness = () => {
  const [value, setValue] = useState(STORED_VALUE);
  return (
    <>
      <NumberInput value={value} onValueChange={setValue} min={0} />
      <output>{value}</output>
    </>
  );
};

describe("NumberInput", () => {
  it("should report the typed number", () => {
    const onValueChange = vi.fn();
    render(<NumberInput value={3} onValueChange={onValueChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "7" } });

    expect(onValueChange).toHaveBeenCalledWith(7);
  });

  it("should commit nothing while the field holds no number", () => {
    const onValueChange = vi.fn();
    render(<NumberInput value={3} onValueChange={onValueChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "" } });

    expect(onValueChange).not.toHaveBeenCalled();
  });

  it("should stay empty once cleared, keeping the stored value untouched", () => {
    render(<StatefulHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue(null);
    expect(screen.getByRole("status")).toHaveTextContent(String(STORED_VALUE));
  });

  it("should accept a fresh number typed into the cleared field", () => {
    render(<StatefulHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.change(input, { target: { value: "5" } });

    expect(input).toHaveValue(5);
    expect(screen.getByRole("status")).toHaveTextContent("5");
  });

  it("should show the stored value again once the cleared field is blurred", () => {
    render(<StatefulHarness />);
    const input = screen.getByRole("spinbutton");

    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);

    expect(input).toHaveValue(STORED_VALUE);
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
