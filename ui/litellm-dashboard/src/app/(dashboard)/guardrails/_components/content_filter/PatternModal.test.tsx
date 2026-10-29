import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PatternModal from "./PatternModal";

describe("PatternModal", () => {
  const mockOnAdd = vi.fn();
  const mockOnCancel = vi.fn();
  const mockOnPatternNameChange = vi.fn();
  const mockOnActionChange = vi.fn();

  const mockPrebuiltPatterns = [
    {
      name: "us_ssn",
      display_name: "US Social Security Number",
      category: "PII Patterns",
      description: "US Social Security Number",
    },
    { name: "email", display_name: "Email address", category: "PII Patterns", description: "Email addresses" },
    { name: "visa", display_name: "Visa card", category: "Financial Patterns", description: "Visa credit cards" },
    {
      name: "aws_access_key",
      display_name: "AWS access key",
      category: "Credential Patterns",
      description: "AWS Access Keys",
    },
  ];

  const mockCategories = ["PII Patterns", "Financial Patterns", "Credential Patterns"];

  const renderModal = () =>
    render(
      <PatternModal
        visible={true}
        prebuiltPatterns={mockPrebuiltPatterns}
        categories={mockCategories}
        selectedPatternName=""
        patternAction="BLOCK"
        onPatternNameChange={mockOnPatternNameChange}
        onActionChange={mockOnActionChange}
        onAdd={mockOnAdd}
        onCancel={mockOnCancel}
      />,
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should show prebuilt pattern options grouped by category and report the picked pattern", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText("Add prebuilt pattern")).toBeInTheDocument();

    await user.click(screen.getAllByRole("combobox")[0]);

    expect(await screen.findByText("PII Patterns")).toBeInTheDocument();
    expect(screen.getByText("Financial Patterns")).toBeInTheDocument();
    expect(screen.getByText("Credential Patterns")).toBeInTheDocument();

    expect(screen.getByText("Email address")).toBeInTheDocument();
    expect(screen.getByText("Visa card")).toBeInTheDocument();
    expect(screen.getByText("AWS access key")).toBeInTheDocument();

    const ssnOptions = await screen.findAllByText("US Social Security Number");
    await user.click(ssnOptions[ssnOptions.length - 1]);

    expect(mockOnPatternNameChange).toHaveBeenCalled();
    expect(mockOnPatternNameChange.mock.calls[0][0]).toBe("us_ssn");
  });

  it("should narrow the pattern options to the typed search text", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText("Add prebuilt pattern")).toBeInTheDocument();

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.keyboard("visa");

    expect(await screen.findByText("Visa card")).toBeInTheDocument();
    expect(screen.queryByText("Email address")).not.toBeInTheDocument();
    expect(screen.queryByText("AWS access key")).not.toBeInTheDocument();
  });

  it("should match the internal pattern name when it is absent from the display name", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText("Add prebuilt pattern")).toBeInTheDocument();

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.keyboard("ssn");

    expect(await screen.findByText("US Social Security Number")).toBeInTheDocument();
    expect(screen.queryByText("Email address")).not.toBeInTheDocument();
    expect(screen.queryByText("Visa card")).not.toBeInTheDocument();
  });

  it("should report the chosen action", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText("Add prebuilt pattern")).toBeInTheDocument();

    await user.click(screen.getAllByRole("combobox")[1]);
    const maskOptions = await screen.findAllByText("Mask");
    await user.click(maskOptions[maskOptions.length - 1]);

    expect(mockOnActionChange).toHaveBeenCalled();
    expect(mockOnActionChange.mock.calls[0][0]).toBe("MASK");
  });

  it("should add and cancel through the footer buttons", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText("Add prebuilt pattern")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Add" }));
    expect(mockOnAdd).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should not render its content when not visible", () => {
    render(
      <PatternModal
        visible={false}
        prebuiltPatterns={mockPrebuiltPatterns}
        categories={mockCategories}
        selectedPatternName=""
        patternAction="BLOCK"
        onPatternNameChange={mockOnPatternNameChange}
        onActionChange={mockOnActionChange}
        onAdd={mockOnAdd}
        onCancel={mockOnCancel}
      />,
    );

    expect(screen.queryByText("Add prebuilt pattern")).not.toBeInTheDocument();
  });

  it("should not raise the dialog above the portalled popup layer its pattern combobox renders into", async () => {
    renderModal();
    await screen.findByText("Add prebuilt pattern");

    const content = document.querySelector('[data-slot="dialog-content"]');
    expect(content).not.toBeNull();
    expect(Array.from(content!.classList).filter((cls) => cls.startsWith("z-"))).toEqual(["z-popup"]);
  });
});
