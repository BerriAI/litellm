import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DeveloperMessageCard from "./DeveloperMessageCard";

vi.mock("../variable_textarea", () => ({
  default: (props: any) => (
    <textarea
      aria-label="Developer message"
      value={props.value}
      onChange={(event) => props.onChange(event.target.value)}
    />
  ),
}));

describe("DeveloperMessageCard", () => {
  it("shows and updates the developer message", () => {
    const onChange = vi.fn();
    render(<DeveloperMessageCard value="Be concise" onChange={onChange} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Developer message" }), { target: { value: "Be kind" } });
    expect(screen.getByText("Optional system instructions for the model")).toBeInTheDocument();
    expect(onChange).toHaveBeenCalledWith("Be kind");
  });
});
