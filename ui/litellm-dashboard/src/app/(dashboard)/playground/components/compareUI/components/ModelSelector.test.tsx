import { render, screen } from "@testing-library/react";
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
    await user.click(await screen.findByRole("option", { name: "gpt-4" }));

    expect(onChange).toHaveBeenCalledWith("gpt-4");
  });

  it("displays a custom value that is not one of the known models", () => {
    render(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} />);

    expect(screen.getByRole("combobox")).toHaveValue("custom-model-123");
  });

  it("reports a custom model typed into the custom name field", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelSelector value="" onChange={onChange} models={MODELS} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "+ Add custom model" }));
    await user.type(await screen.findByPlaceholderText("Custom Model Name (Enter to add)"), "my-custom-model{Enter}");

    expect(onChange).toHaveBeenCalledWith("my-custom-model");
  });

  it("disables the control when disabled is set", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { rerender } = render(<ModelSelector value="custom-model-123" onChange={onChange} models={MODELS} />);
    expect(screen.getByRole("combobox")).toBeEnabled();

    rerender(<ModelSelector value="custom-model-123" onChange={onChange} models={MODELS} disabled={true} />);

    const combobox = screen.getByRole("combobox");
    expect(combobox).toBeDisabled();

    await user.click(combobox);
    await user.keyboard("gpt-4");

    expect(combobox).toHaveValue("custom-model-123");
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
