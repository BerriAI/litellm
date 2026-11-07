import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StorePromptsMetadataToggle } from "./StorePromptsToggle";

describe("StorePromptsMetadataToggle", () => {
  it("leaves an unset default untouched and writes false while preserving metadata", () => {
    const onChange = vi.fn();
    render(<StorePromptsMetadataToggle value={'{"note":"keep"}'} onChange={onChange} />);
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("switch", { name: "Store prompts in spend logs" }));
    expect(JSON.parse(onChange.mock.calls[0][0])).toEqual({ note: "keep", store_prompts_in_spend_logs: false });
  });

  it("can remove an existing restriction", () => {
    const onChange = vi.fn();
    render(<StorePromptsMetadataToggle value={'{"store_prompts_in_spend_logs":false}'} onChange={onChange} />);
    expect(screen.getByRole("switch")).not.toBeChecked();
    fireEvent.click(screen.getByRole("switch"));
    expect(JSON.parse(onChange.mock.calls[0][0])).toEqual({ store_prompts_in_spend_logs: true });
  });

  it("shows a saved string false restriction as off", () => {
    render(<StorePromptsMetadataToggle value={'{"store_prompts_in_spend_logs":"False"}'} onChange={vi.fn()} />);
    expect(screen.getByRole("switch", { name: "Store prompts in spend logs" })).not.toBeChecked();
  });

  it("does not overwrite invalid metadata", () => {
    const onChange = vi.fn();
    render(<StorePromptsMetadataToggle value="{" onChange={onChange} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(screen.getByRole("switch"));
    expect(onChange).not.toHaveBeenCalled();
  });
});
