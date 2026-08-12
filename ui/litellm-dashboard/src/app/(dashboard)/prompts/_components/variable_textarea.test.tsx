import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import VariableTextArea from "./variable_textarea";

describe("VariableTextArea", () => {
  it("updates text and identifies template variables", () => {
    const onChange = vi.fn();
    render(<VariableTextArea value="Hello {{name}}" onChange={onChange} placeholder="Prompt" />);
    expect(screen.getByText("Detected variables:")).toBeInTheDocument();
    expect(screen.getByText("name")).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("Prompt"), { target: { value: "Hi" } });
    expect(onChange).toHaveBeenCalledWith("Hi");
  });
});
