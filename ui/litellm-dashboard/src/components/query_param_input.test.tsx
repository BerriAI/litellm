import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import QueryParamInput from "./query_param_input";

describe("QueryParamInput", () => {
  it("should render", () => {
    render(<QueryParamInput />);
    expect(screen.getByRole("button", { name: /add query parameter/i })).toBeInTheDocument();
  });

  it("should render existing pairs from initial value", () => {
    render(<QueryParamInput value={{ version: "v1" }} />);
    expect(screen.getByDisplayValue("version")).toBeInTheDocument();
    expect(screen.getByDisplayValue("v1")).toBeInTheDocument();
  });

  it("should add a new empty pair when clicking Add Query Parameter", async () => {
    const user = userEvent.setup();
    render(<QueryParamInput />);

    await user.click(screen.getByRole("button", { name: /add query parameter/i }));

    expect(screen.getByPlaceholderText(/parameter name/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/parameter value/i)).toBeInTheDocument();
  });

  it("should call onChange when a value is typed", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<QueryParamInput value={{}} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /add query parameter/i }));
    await user.type(screen.getByPlaceholderText(/parameter name/i), "limit");

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ limit: "" })
    );
  });

  it("should remove a pair when clicking the remove icon", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<QueryParamInput value={{ foo: "bar" }} onChange={onChange} />);

    await user.click(screen.getByRole("img", { name: /minus-circle/i }));

    expect(screen.queryByDisplayValue("foo")).not.toBeInTheDocument();
    expect(onChange).toHaveBeenCalledWith({});
  });
});
