import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModelSelector } from "./ModelSelector";

const MODELS = ["gpt-4", "gpt-3.5-turbo"];

describe("ModelSelector", () => {
  it("should render", () => {
    render(<ModelSelector value="" onChange={vi.fn()} models={MODELS} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("reports the model the user picks from the dropdown", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelSelector value="" onChange={onChange} models={MODELS} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByTitle("gpt-4"));

    expect(onChange).toHaveBeenCalledWith("gpt-4");
  });

  it("displays a custom value that is not one of the known models", () => {
    render(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} />);

    expect(screen.getByTitle("custom-model-123")).toHaveTextContent("custom-model-123");
  });

  it("reports a custom model typed into the custom name field", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelSelector value="" onChange={onChange} models={MODELS} />);

    await user.click(screen.getByRole("combobox"));
    fireEvent.click(await screen.findByTitle("+ Add custom model"));
    await user.type(await screen.findByPlaceholderText("Custom Model Name (Enter to add)"), "my-custom-model{Enter}");

    expect(onChange).toHaveBeenCalledWith("my-custom-model");
  });

  it("disables the control when disabled is set", () => {
    const { rerender } = render(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} />);
    expect(screen.getByRole("combobox")).toBeEnabled();

    rerender(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} disabled={true} />);
    expect(screen.getByRole("combobox")).toBeDisabled();
  });
});
