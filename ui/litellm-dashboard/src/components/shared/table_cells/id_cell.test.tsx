import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { IdCell } from "./id_cell";

const { copyToClipboardMock } = vi.hoisted(() => ({ copyToClipboardMock: vi.fn() }));

vi.mock("@/utils/dataUtils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/dataUtils")>()),
  copyToClipboard: copyToClipboardMock,
}));

describe("IdCell", () => {
  it("renders '-' for empty values", () => {
    render(<IdCell value={null} />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("renders the custom fallback for empty values", () => {
    render(<IdCell value="" fallback="—" />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders the full id as a non-interactive pill by default", () => {
    render(<IdCell value="sk-1234567890abcdef" />);
    const el = screen.getByText("sk-1234567890abcdef");
    expect(el.tagName).toBe("SPAN");
    expect(el.className).toContain("bg-blue-50");
    expect(el.className).toContain("font-mono");
    expect(el.className).toContain("max-w-[15ch]");
    expect(el.className).toContain("truncate");
  });

  it("renders plain mono text without pill styling for the plain variant", () => {
    render(<IdCell value="req-123" variant="plain" />);
    const el = screen.getByText("req-123");
    expect(el.className).toContain("font-mono");
    expect(el.className).not.toContain("bg-blue-50");
  });

  it("does not truncate when truncate is false", () => {
    render(<IdCell value="audit-object-id" truncate={false} />);
    expect(screen.getByText("audit-object-id").className).not.toContain("truncate");
  });

  it("becomes a button that fires onClick with the id value", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<IdCell value="team-42" onClick={onClick} />);
    await user.click(screen.getByRole("button", { name: "team-42" }));
    expect(onClick).toHaveBeenCalledWith("team-42");
  });

  it("stays non-interactive when disabled, even with onClick", () => {
    const onClick = vi.fn();
    render(<IdCell value="tag-1" onClick={onClick} disabled />);
    expect(screen.queryByRole("button", { name: "tag-1" })).not.toBeInTheDocument();
  });

  it("copies the id via the trailing copy button without triggering row clicks", async () => {
    const user = userEvent.setup();
    const rowClick = vi.fn();
    render(
      <div onClick={rowClick}>
        <IdCell value="key-hash-9" copyable />
      </div>,
    );
    await user.click(screen.getByRole("button", { name: "Copy ID" }));
    expect(copyToClipboardMock).toHaveBeenCalledWith("key-hash-9");
    expect(rowClick).not.toHaveBeenCalled();
  });

  it("passes dataTestId through to the id element", () => {
    render(<IdCell value="k-1" dataTestId="key-id-cell" />);
    expect(screen.getByTestId("key-id-cell")).toHaveTextContent("k-1");
  });
});
