import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import KeyValueInput from "./key_value_input";

describe("KeyValueInput", () => {
  it("should render", () => {
    render(<KeyValueInput />);
    expect(screen.getByRole("button", { name: /add header/i })).toBeInTheDocument();
  });

  it("should render existing key-value pairs from initial value", () => {
    render(<KeyValueInput value={{ "X-Api-Key": "secret123" }} />);
    expect(screen.getByDisplayValue("X-Api-Key")).toBeInTheDocument();
    expect(screen.getByDisplayValue("secret123")).toBeInTheDocument();
  });

  it("should add a new empty pair when clicking Add Header", async () => {
    const user = userEvent.setup();
    render(<KeyValueInput />);

    await user.click(screen.getByRole("button", { name: /add header/i }));

    expect(screen.getByPlaceholderText("Header Name")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Header Value")).toBeInTheDocument();
  });

  it("should call onChange when a key is typed", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<KeyValueInput value={{}} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /add header/i }));
    await user.type(screen.getByPlaceholderText("Header Name"), "Authorization");

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ Authorization: "" })
    );
  });

  it("should remove a pair when clicking the remove icon", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<KeyValueInput value={{ key1: "val1" }} onChange={onChange} />);

    expect(screen.getByDisplayValue("key1")).toBeInTheDocument();

    // MinusCircleOutlined renders with role="img"
    await user.click(screen.getByRole("img", { name: /minus-circle/i }));

    expect(screen.queryByDisplayValue("key1")).not.toBeInTheDocument();
    expect(onChange).toHaveBeenCalledWith({});
  });
});
