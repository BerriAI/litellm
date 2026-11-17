import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import KeywordModal from "./KeywordModal";

describe("KeywordModal", () => {
  const handlers = {
    onKeywordChange: vi.fn(),
    onActionChange: vi.fn(),
    onDescriptionChange: vi.fn(),
    onAdd: vi.fn(),
    onCancel: vi.fn(),
  };

  const renderModal = (overrides: Partial<React.ComponentProps<typeof KeywordModal>> = {}) =>
    render(<KeywordModal visible keyword="" action="BLOCK" description="" {...handlers} {...overrides} />);

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the keyword, action and description fields", async () => {
    renderModal();

    expect(await screen.findByText("Add blocked keyword")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter sensitive keyword or phrase")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Explain why this keyword is sensitive")).toBeInTheDocument();
    expect(screen.getByText("Description (optional)")).toBeInTheDocument();
    expect(
      screen.getByText("Choose what action the guardrail should take when this keyword is detected"),
    ).toBeInTheDocument();
  });

  it("should report keyword edits", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(await screen.findByPlaceholderText("Enter sensitive keyword or phrase"), {
      target: { value: "s" },
    });

    expect(handlers.onKeywordChange).toHaveBeenCalledWith("s");
  });

  it("should report description edits", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(await screen.findByPlaceholderText("Explain why this keyword is sensitive"), {
      target: { value: "x" },
    });

    expect(handlers.onDescriptionChange).toHaveBeenCalledWith("x");
  });

  it("should report the chosen action", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole("combobox"));
    const maskOptions = await screen.findAllByText("Mask");
    await user.click(maskOptions[maskOptions.length - 1]);

    expect(handlers.onActionChange).toHaveBeenCalled();
    expect(handlers.onActionChange.mock.calls[0][0]).toBe("MASK");
  });

  it("should show the current keyword and description values", async () => {
    renderModal({ keyword: "secret", description: "sensitive term" });

    expect(await screen.findByDisplayValue("secret")).toBeInTheDocument();
    expect(screen.getByDisplayValue("sensitive term")).toBeInTheDocument();
  });

  it("should add and cancel through the footer buttons", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole("button", { name: "Add" }));
    expect(handlers.onAdd).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(handlers.onCancel).toHaveBeenCalledTimes(1);
  });

  it("should not render its content when not visible", () => {
    renderModal({ visible: false });

    expect(screen.queryByText("Add blocked keyword")).not.toBeInTheDocument();
  });

  it("should not raise the dialog above the portalled popup layer its Action select renders into", async () => {
    renderModal();
    await screen.findByText("Add blocked keyword");

    const content = document.querySelector('[data-slot="dialog-content"]');
    expect(content).not.toBeNull();
    expect(Array.from(content!.classList).filter((cls) => cls.startsWith("z-"))).toEqual(["z-popup"]);
  });
});
