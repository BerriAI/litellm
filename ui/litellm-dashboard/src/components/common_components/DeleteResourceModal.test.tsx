import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import DeleteResourceModal from "./DeleteResourceModal";

describe("DeleteResourceModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnOk = vi.fn();

  const defaultProps = {
    isOpen: true,
    title: "Delete Resource",
    message: "Are you sure you want to delete this resource?",
    onCancel: mockOnCancel,
    onOk: mockOnOk,
    confirmLoading: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    expect(screen.getByText("Delete Resource")).toBeInTheDocument();
  });

  it("should render the title correctly", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} title="Custom Delete Title" />);
    expect(screen.getByText("Custom Delete Title")).toBeInTheDocument();
  });

  it("should render the message correctly", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} message="This is a custom message" />);
    expect(screen.getByText("This is a custom message")).toBeInTheDocument();
  });

  it("should render alert message when provided", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} alertMessage="Warning: This action cannot be undone" />);
    expect(screen.getByText("Warning: This action cannot be undone")).toBeInTheDocument();
  });

  it("should not render alert message when not provided", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should render resourceInformation and resourceInformationTitle correctly", () => {
    const resourceInformation = [
      { label: "Name", value: "Test Resource" },
      { label: "ID", value: "123" },
    ];
    renderWithProviders(
      <DeleteResourceModal
        {...defaultProps}
        resourceInformationTitle="Resource Details"
        resourceInformation={resourceInformation}
      />,
    );
    expect(screen.getByText("Resource Details")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Test Resource")).toBeInTheDocument();
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("123")).toBeInTheDocument();
  });

  it("should render dash for null or undefined resource information values", () => {
    const resourceInformation = [
      { label: "Name", value: null },
      { label: "ID", value: undefined },
      { label: "Status", value: "Active" },
    ];
    renderWithProviders(
      <DeleteResourceModal {...defaultProps} resourceInformation={resourceInformation} />,
    );
    expect(screen.getAllByText("-")).toHaveLength(2);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("should render resource information with number values", () => {
    const resourceInformation = [{ label: "Count", value: 42 }];
    renderWithProviders(<DeleteResourceModal {...defaultProps} resourceInformation={resourceInformation} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelButton);
    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should call onOk when delete button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    const deleteButton = screen.getByRole("button", { name: /delete/i });
    await user.click(deleteButton);
    expect(mockOnOk).toHaveBeenCalledTimes(1);
  });

  it("should disable delete button when requiredConfirmation is not entered", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />);
    const deleteButton = screen.getByRole("button", { name: /delete/i });
    expect(deleteButton).toBeDisabled();
  });

  it("should disable delete button when requiredConfirmation input does not match exactly", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />);
    const input = screen.getByPlaceholderText("DELETE");
    await user.type(input, "DELET");
    const deleteButton = screen.getByRole("button", { name: /delete/i });
    expect(deleteButton).toBeDisabled();
  });

  it("should enable delete button when requiredConfirmation input matches exactly", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />);
    const input = screen.getByPlaceholderText("DELETE");
    await user.type(input, "DELETE");
    const deleteButton = screen.getByRole("button", { name: /delete/i });
    expect(deleteButton).not.toBeDisabled();
  });

  it("should reset requiredConfirmation input when modal opens", async () => {
    const user = userEvent.setup();
    const { rerender } = renderWithProviders(
      <DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />,
    );
    const input = screen.getByPlaceholderText("DELETE");
    await user.type(input, "DELETE");
    expect(input).toHaveValue("DELETE");

    rerender(<DeleteResourceModal {...defaultProps} isOpen={false} requiredConfirmation="DELETE" />);
    rerender(<DeleteResourceModal {...defaultProps} isOpen={true} requiredConfirmation="DELETE" />);

    const newInput = screen.getByPlaceholderText("DELETE");
    expect(newInput).toHaveValue("");
  });

  it("should display deleting text on delete button when confirmLoading is true", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} confirmLoading={true} />);
    expect(screen.getByText("Deleting...")).toBeInTheDocument();
  });

  it("should display delete text on delete button when confirmLoading is false", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} confirmLoading={false} />);
    const deleteButton = screen.getByRole("button", { name: /delete/i });
    expect(deleteButton).toBeInTheDocument();
    expect(screen.queryByText("Deleting...")).not.toBeInTheDocument();
  });

  it("should disable delete button when confirmLoading is true", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} confirmLoading={true} />);
    const deleteButton = screen.getByText("Deleting...").closest("button");
    expect(deleteButton).toBeDisabled();
  });

  it("should disable cancel button when confirmLoading is true", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} confirmLoading={true} />);
    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should disable delete button when confirmLoading is true even if requiredConfirmation matches", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <DeleteResourceModal {...defaultProps} confirmLoading={true} requiredConfirmation="DELETE" />,
    );
    const input = screen.getByPlaceholderText("DELETE");
    await user.type(input, "DELETE");
    const deleteButton = screen.getByText("Deleting...").closest("button");
    expect(deleteButton).toBeDisabled();
  });

  it("should render required confirmation prompt with correct text", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />);
    expect(screen.getByText(/Type/i)).toBeInTheDocument();
    expect(screen.getByText("DELETE")).toBeInTheDocument();
    expect(screen.getByText(/to confirm deletion/i)).toBeInTheDocument();
  });

  it("should not render required confirmation section when not provided", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    expect(screen.queryByPlaceholderText("DELETE")).not.toBeInTheDocument();
  });

  it("should not render modal when isOpen is false", () => {
    renderWithProviders(<DeleteResourceModal {...defaultProps} isOpen={false} />);
    expect(screen.queryByText("Delete Resource")).not.toBeInTheDocument();
  });
});
