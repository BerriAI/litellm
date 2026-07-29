import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModelSelector } from "./ModelSelector";

describe("ModelSelector", () => {
  it("should render", () => {
    const onChange = vi.fn();
    const models = ["gpt-4", "gpt-3.5-turbo"];
    const { container } = render(<ModelSelector value="" onChange={onChange} models={models} />);
    const select = container.querySelector(".ant-select");
    expect(select).toBeInTheDocument();
  });

  it("allows selecting a model and displays custom values", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const models = ["gpt-4", "gpt-3.5-turbo"];
    const { container } = render(<ModelSelector value="" onChange={onChange} models={models} />);

    const select = container.querySelector(".ant-select-selector") as HTMLElement;
    await user.click(select);

    await waitFor(() => {
      const gpt4Option = document.querySelector('[title="gpt-4"].ant-select-item-option') as HTMLElement;
      expect(gpt4Option).toBeInTheDocument();
    });

    const gpt4Option = document.querySelector('[title="gpt-4"].ant-select-item-option') as HTMLElement;
    await user.click(gpt4Option);
    expect(onChange).toHaveBeenCalledWith("gpt-4");

    const { container: container2, rerender } = render(
      <ModelSelector value="custom-model-123" onChange={onChange} models={models} />,
    );
    const selectedValue = container2.querySelector(".ant-select-selection-item");
    expect(selectedValue).toHaveTextContent("custom-model-123");

    rerender(<ModelSelector value="custom-model-123" onChange={onChange} models={models} disabled={true} />);
    const selectElement = container2.querySelector(".ant-select");
    expect(selectElement).toHaveClass("ant-select-disabled");
  });
});
