import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import KeyValueInput from "./key_value_input";

describe("KeyValueInput", () => {
  it("renders existing header pairs", () => {
    render(<KeyValueInput value={{ Authorization: "Bearer token" }} />);

    expect(screen.getByPlaceholderText("Header Name")).toHaveValue("Authorization");
    expect(screen.getByPlaceholderText("Header Value")).toHaveValue("Bearer token");
  });

  it("adds a header row and emits edits", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<KeyValueInput onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /Add Header$/ }));
    fireEvent.change(screen.getByPlaceholderText("Header Name"), { target: { value: "X-Trace" } });
    fireEvent.change(screen.getByPlaceholderText("Header Value"), { target: { value: "enabled" } });

    expect(onChange).toHaveBeenLastCalledWith({ "X-Trace": "enabled" });
  });
});
