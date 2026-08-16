import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import QueryParamInput from "./query_param_input";

describe("QueryParamInput", () => {
  it("renders existing query parameter pairs", () => {
    render(<QueryParamInput value={{ version: "v1" }} />);

    expect(screen.getByPlaceholderText("Parameter Name (e.g., version)")).toHaveValue("version");
    expect(screen.getByPlaceholderText("Parameter Value (e.g., v1)")).toHaveValue("v1");
  });

  it("adds a query parameter row and emits edits", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<QueryParamInput onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /Add Query Parameter$/ }));
    await user.type(screen.getByPlaceholderText("Parameter Name (e.g., version)"), "region");
    await user.type(screen.getByPlaceholderText("Parameter Value (e.g., v1)"), "us-west");

    expect(onChange).toHaveBeenLastCalledWith({ region: "us-west" });
  });
});
