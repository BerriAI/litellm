import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ToolModal from "./tool_modal";

describe("ToolModal", () => {
  it("rejects invalid JSON and saves valid JSON", async () => {
    const onSave = vi.fn();
    render(<ToolModal visible initialJson="{}" onSave={onSave} onClose={vi.fn()} />);
    expect(await screen.findByRole("dialog")).toHaveClass("max-h-[calc(100dvh-2rem)]", "overflow-y-auto");
    const editor = await screen.findByPlaceholderText("Paste your tool JSON here...");
    fireEvent.change(editor, { target: { value: "invalid" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByText(/invalid json format/i)).toBeInTheDocument();
    fireEvent.change(editor, { target: { value: '{"type":"function"}' } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(onSave).toHaveBeenCalledWith('{"type":"function"}');
  });
});
