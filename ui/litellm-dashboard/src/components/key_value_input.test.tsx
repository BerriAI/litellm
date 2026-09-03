import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import KeyValueInput, { type KeyValuePair } from "./key_value_input";

const ControlledKeyValueInput = ({ onChange }: { onChange?: (value: readonly KeyValuePair[]) => void }) => {
  const [value, setValue] = useState<readonly KeyValuePair[]>([]);
  return (
    <KeyValueInput
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange?.(next);
      }}
    />
  );
};

describe("KeyValueInput", () => {
  it("renders existing header pairs", () => {
    render(<KeyValueInput value={[["Authorization", "Bearer token"]]} />);

    expect(screen.getByPlaceholderText("Header Name")).toHaveValue("Authorization");
    expect(screen.getByPlaceholderText("Header Value")).toHaveValue("Bearer token");
  });

  it("adds a header row and emits edits", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ControlledKeyValueInput onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /Add Header$/ }));
    fireEvent.change(screen.getByPlaceholderText("Header Name"), { target: { value: "X-Trace" } });
    fireEvent.change(screen.getByPlaceholderText("Header Value"), { target: { value: "enabled" } });

    expect(onChange).toHaveBeenLastCalledWith([["X-Trace", "enabled"]]);
  });

  it("renders no rows once the value prop goes back to empty", () => {
    const { rerender } = render(<KeyValueInput value={[["Authorization", "Bearer token"]]} />);
    expect(screen.getByPlaceholderText("Header Name")).toHaveValue("Authorization");

    rerender(<KeyValueInput value={[]} />);

    expect(screen.queryByPlaceholderText("Header Name")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Header Value")).not.toBeInTheDocument();
  });

  it("keeps a half-typed row editable while its name is still empty", async () => {
    const user = userEvent.setup();
    render(<ControlledKeyValueInput />);

    await user.click(screen.getByRole("button", { name: /Add Header$/ }));
    await user.click(screen.getByRole("button", { name: /Add Header$/ }));
    expect(screen.getAllByPlaceholderText("Header Name")).toHaveLength(2);

    fireEvent.change(screen.getAllByPlaceholderText("Header Value")[1], { target: { value: "pending" } });

    expect(screen.getAllByPlaceholderText("Header Name")).toHaveLength(2);
    expect(screen.getAllByPlaceholderText("Header Value")[1]).toHaveValue("pending");
  });

  it("removes only the row whose remove button is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <KeyValueInput
        value={[
          ["a", "1"],
          ["b", "2"],
        ]}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Remove header 1" }));

    expect(onChange).toHaveBeenLastCalledWith([["b", "2"]]);
  });
});
