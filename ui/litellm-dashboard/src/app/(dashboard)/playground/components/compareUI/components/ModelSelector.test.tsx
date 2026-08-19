import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModelSelector } from "./ModelSelector";

const MODELS = ["gpt-4", "gpt-3.5-turbo"];

const openList = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("combobox"));
};

describe("ModelSelector", () => {
  it("should render", () => {
    render(<ModelSelector value="" onChange={vi.fn()} models={MODELS} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("reports the model the user picks from the dropdown", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelSelector value="" onChange={onChange} models={MODELS} />);

    await openList(user);
    const matches = await screen.findAllByText("gpt-4");
    await user.click(matches[matches.length - 1]);

    await waitFor(() => expect(onChange).toHaveBeenCalledWith("gpt-4"));
  });

  it("displays a custom value that is not one of the known models", () => {
    render(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} />);

    expect(screen.getByRole("combobox")).toHaveValue("custom-model-123");
  });

  it("disables the control when disabled is set", () => {
    const { rerender } = render(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} />);
    expect(screen.getByRole("combobox")).toBeEnabled();

    rerender(<ModelSelector value="custom-model-123" onChange={vi.fn()} models={MODELS} disabled={true} />);
    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("commits a typed custom model on Enter", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModelSelector value="" onChange={onChange} models={MODELS} />);

    await openList(user);
    const customOption = await screen.findAllByText("+ Add custom model");
    await user.click(customOption[customOption.length - 1]);

    const customInput = await screen.findByPlaceholderText("Custom Model Name (Enter to add)");
    await user.type(customInput, "my-finetune");
    await user.keyboard("{Enter}");

    await waitFor(() => expect(onChange).toHaveBeenCalledWith("my-finetune"));
  });
});
