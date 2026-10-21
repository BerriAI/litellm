import { act, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";
import { copyToClipboard } from "@/utils/dataUtils";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import CopyButton from "./CopyButton";

vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn() }));
vi.mock("lucide-react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("lucide-react")>()),
  Copy: (props: ComponentProps<"svg">) => <svg {...props} role="img" aria-label="Copy icon" />,
  Check: (props: ComponentProps<"svg">) => <svg {...props} role="img" aria-label="Copied icon" />,
}));

const clipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, "clipboard");

afterEach(() => {
  vi.useRealTimers();
  vi.resetAllMocks();
  if (clipboardDescriptor) Object.defineProperty(navigator, "clipboard", clipboardDescriptor);
  else Reflect.deleteProperty(navigator, "clipboard");
});

describe("CopyButton", () => {
  it("renders nothing when there is no value", () => {
    renderWithProviders(<CopyButton value={null} label="Copy value" />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("shows the confirmation checkmark only after a successful write", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();

    await user.click(button);

    expect(writeText).toHaveBeenCalledWith("secret-value");
    expect(await screen.findByRole("img", { name: "Copied icon" })).toBeInTheDocument();
  });

  it("does not show the checkmark when the clipboard write is rejected", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("permission denied"));
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    await user.click(button);

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("secret-value"));
    expect(screen.queryByRole("img", { name: "Copied icon" })).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
  });

  it("does not show the checkmark when the clipboard API is unavailable", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });

    renderWithProviders(<CopyButton value="secret-value" label="Copy value" />);

    const button = screen.getByRole("button", { name: "Copy value" });
    await user.click(button);

    expect(screen.queryByRole("img", { name: "Copied icon" })).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
    expect(copyToClipboard).not.toHaveBeenCalled();
  });

  it("keeps the default icon styling, custom classes, title and 1200ms confirmation", async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    renderWithProviders(
      <CopyButton value="request-id" label="Copy request ID" className="shrink-0" iconClassName="size-3" />,
    );
    const button = screen.getByRole("button", { name: "Copy request ID" });
    expect(button).toHaveAttribute("title", "Copy request ID");
    expect(button).toHaveClass("hover:text-primary", "shrink-0");
    expect(button).not.toHaveTextContent(/.+/);
    expect(screen.getByRole("img", { name: "Copy icon" })).toHaveClass("size-3");

    await act(async () => fireEvent.click(button));
    expect(writeText).toHaveBeenCalledWith("request-id");
    expect(copyToClipboard).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1199));
    expect(screen.getByRole("img", { name: "Copied icon" })).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
  });

  it("shows an outlined action with custom confirmation text for 1600ms", async () => {
    vi.useFakeTimers();
    vi.mocked(copyToClipboard).mockResolvedValue(true);
    renderWithProviders(
      <CopyButton variant="action" value="step-context" label="Copy step" copiedLabel="Step copied" />,
    );
    const button = screen.getByRole("button", { name: "Copy step" });
    expect(button).toHaveTextContent("Copy step");
    expect(button).toHaveClass("border", "gap-1.5", "hover:text-foreground");
    expect(screen.getByRole("img", { name: "Copy icon" })).toHaveClass("size-3");

    await act(async () => fireEvent.click(button));
    expect(copyToClipboard).toHaveBeenCalledWith("step-context", "Step copied");
    expect(button).toHaveTextContent("Step copied");
    expect(screen.getByRole("img", { name: "Copied icon" })).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1599));
    expect(button).toHaveTextContent("Step copied");
    act(() => vi.advanceTimersByTime(1));
    expect(button).toHaveTextContent("Copy step");
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
  });

  it("shows a tooltip and no text for an icon-only action", async () => {
    const user = userEvent.setup();
    vi.mocked(copyToClipboard).mockResolvedValue(true);
    renderWithProviders(<CopyButton variant="action" value="tool-result" label="Copy tool result" iconOnly />);
    const button = screen.getByRole("button", { name: "Copy tool result" });
    expect(button).not.toHaveTextContent(/.+/);
    expect(button).not.toHaveAttribute("title");
    await user.hover(button);
    expect(await screen.findByText("Copy tool result")).toBeVisible();
    await user.click(button);
    expect(copyToClipboard).toHaveBeenCalledWith("tool-result", "Copied");
    await user.unhover(button);
    await user.hover(button);
    expect(await screen.findByText("Copied")).toBeVisible();
  });

  it("keeps the action label and copy icon when copying fails", async () => {
    const user = userEvent.setup();
    vi.mocked(copyToClipboard).mockResolvedValue(false);
    renderWithProviders(<CopyButton variant="action" value="step-context" label="Copy step" />);
    const button = screen.getByRole("button", { name: "Copy step" });
    await user.click(button);
    expect(copyToClipboard).toHaveBeenCalledWith("step-context", "Copied");
    expect(button).toHaveTextContent("Copy step");
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Copied icon" })).not.toBeInTheDocument();
  });

  it("retains an empty action while default empty values remain hidden", async () => {
    const user = userEvent.setup();
    vi.mocked(copyToClipboard).mockResolvedValue(false);
    renderWithProviders(
      <>
        <CopyButton value="" label="Copy value" />
        <CopyButton variant="action" value="" label="Copy empty result" iconOnly />
      </>,
    );
    expect(screen.queryByRole("button", { name: "Copy value" })).not.toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Copy empty result" });
    await user.click(button);
    expect(copyToClipboard).toHaveBeenCalledWith("", "Copied");
    expect(screen.getByRole("img", { name: "Copy icon" })).toBeInTheDocument();
  });
});
