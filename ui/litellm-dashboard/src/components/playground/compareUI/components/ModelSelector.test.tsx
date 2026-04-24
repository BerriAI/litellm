import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModelSelector } from "./ModelSelector";

describe("ModelSelector", () => {
  it("should render", () => {
    const onChange = vi.fn();
    const models = ["gpt-4", "gpt-3.5-turbo"];
    render(<ModelSelector value="" onChange={onChange} models={models} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("allows selecting a model and displays custom values", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const models = ["gpt-4", "gpt-3.5-turbo"];
    render(<ModelSelector value="" onChange={onChange} models={models} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    const gpt4Option = await screen.findByRole("option", { name: "gpt-4" });
    await user.click(gpt4Option);
    expect(onChange).toHaveBeenCalledWith("gpt-4");

    // Re-render with a value not in the model list; it should still be shown.
    const { rerender } = render(
      <ModelSelector value="custom-model-123" onChange={onChange} models={models} />,
    );
    const comboboxes = screen.getAllByRole("combobox");
    expect(comboboxes.at(-1)).toHaveTextContent("custom-model-123");

    rerender(<ModelSelector value="custom-model-123" onChange={onChange} models={models} disabled={true} />);
    const disabledCombobox = screen.getAllByRole("combobox").at(-1)!;
    expect(disabledCombobox).toBeDisabled();
  });
});
