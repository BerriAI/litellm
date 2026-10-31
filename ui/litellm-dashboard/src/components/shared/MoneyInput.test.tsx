import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { MoneyInput } from "./MoneyInput";

vi.mock("@/app/(dashboard)/hooks/uiSettings/useCurrency", () => ({
  useCurrency: () => ({ code: "EUR", usd_rate: 0.92 }),
}));

describe("MoneyInput", () => {
  it("shows EUR, emits USD and keeps entered values on round trip", () => {
    const onChange = vi.fn();
    function Form() {
      const [value, setValue] = useState<number | null>(null);
      return (
        <MoneyInput
          aria-label="Budget"
          value={value}
          onChange={(next) => {
            onChange(next);
            setValue(next);
          }}
        />
      );
    }
    render(<Form />);
    const input = screen.getByRole("textbox", { name: "Budget" });
    expect(screen.getByText("€")).toBeInTheDocument();
    expect(input).toHaveValue("");
    fireEvent.change(input, { target: { value: "100" } });
    expect(onChange).toHaveBeenLastCalledWith(100 / 0.92);
    expect(input).toHaveValue("100");
    fireEvent.blur(input);
    expect(input).toHaveValue("100");
    fireEvent.change(input, { target: { value: "" } });
    expect(onChange).toHaveBeenLastCalledWith(null);
    expect(input).toHaveValue("");
  });

  it("keeps an incomplete input without emitting an invalid amount", () => {
    const onChange = vi.fn();
    render(<MoneyInput aria-label="Budget" value={1} onChange={onChange} />);
    const input = screen.getByRole("textbox", { name: "Budget" });
    fireEvent.change(input, { target: { value: "-" } });
    expect(input).toHaveValue("-");
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.blur(input);
    expect(input).toHaveValue("0.92");
    fireEvent.change(input, { target: { value: `17${"0".repeat(307)}` } });
    expect(onChange).not.toHaveBeenCalled();
  });
});
