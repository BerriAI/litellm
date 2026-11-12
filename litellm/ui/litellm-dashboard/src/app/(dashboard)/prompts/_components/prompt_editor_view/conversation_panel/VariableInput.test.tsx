import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import VariableInput from "./VariableInput";

describe("VariableInput", () => {
  it("updates a named template variable", () => {
    const onVariableChange = vi.fn();
    render(
      <VariableInput extractedVariables={["name"]} variables={{ name: "Ada" }} onVariableChange={onVariableChange} />,
    );
    fireEvent.change(screen.getByPlaceholderText("Enter value for name"), { target: { value: "Grace" } });
    expect(onVariableChange).toHaveBeenCalledWith("name", "Grace");
  });
});
