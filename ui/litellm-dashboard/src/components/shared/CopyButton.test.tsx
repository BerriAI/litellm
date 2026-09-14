import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import CopyButton from "./CopyButton";

describe("CopyButton", () => {
  it("renders nothing when there is no value", () => {
    const { container } = renderWithProviders(<CopyButton value={null} label="Copy value" />);
    expect(container.querySelector("button")).toBeNull();
  });

  it("shows the confirmation checkmark only after a successful write", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    expect(button.querySelector(".lucide-copy")).toBeInTheDocument();

    await user.click(button);

    expect(writeText).toHaveBeenCalledWith("secret-value");
    await waitFor(() => expect(button.querySelector(".lucide-check")).toBeInTheDocument());
  });

  it("does not show the checkmark when the clipboard write is rejected", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("permission denied"));
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    await user.click(button);

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("secret-value"));
    expect(button.querySelector(".lucide-check")).not.toBeInTheDocument();
    expect(button.querySelector(".lucide-copy")).toBeInTheDocument();
  });

  it("does not show the checkmark when the clipboard API is unavailable", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    await user.click(button);

    expect(button.querySelector(".lucide-check")).not.toBeInTheDocument();
    expect(button.querySelector(".lucide-copy")).toBeInTheDocument();
  });
});
