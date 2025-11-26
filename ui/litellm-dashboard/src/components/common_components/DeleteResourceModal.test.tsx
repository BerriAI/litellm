import { describe, it, expect, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import DeleteResourceModal from "./DeleteResourceModal";

describe("DeleteResourceModal", () => {
  const defaultProps = {
    isOpen: true,
    title: "Delete Resource",
    message: "Are you sure you want to delete this resource?",
    onCancel: vi.fn(),
    onOk: vi.fn(),
    confirmLoading: false,
  };

  it("renders", () => {
    const { getByText } = renderWithProviders(<DeleteResourceModal {...defaultProps} />);
    expect(getByText("Delete Resource")).toBeInTheDocument();
  });

  it("renders the title correctly", () => {
    const { getByText } = renderWithProviders(<DeleteResourceModal {...defaultProps} title="Custom Delete Title" />);
    expect(getByText("Custom Delete Title")).toBeInTheDocument();
  });

  it("renders the message correctly", () => {
    const { getByText } = renderWithProviders(
      <DeleteResourceModal {...defaultProps} message="This is a custom message" />,
    );
    expect(getByText("This is a custom message")).toBeInTheDocument();
  });

  it("renders the resourceInformation and resourceInformationTitle correctly", () => {
    const resourceInformation = [
      { label: "Name", value: "Test Resource" },
      { label: "ID", value: "123" },
    ];
    const { getByText } = renderWithProviders(
      <DeleteResourceModal
        {...defaultProps}
        resourceInformationTitle="Resource Details"
        resourceInformation={resourceInformation}
      />,
    );
    expect(getByText("Resource Details")).toBeInTheDocument();
    expect(getByText("Name")).toBeInTheDocument();
    expect(getByText("Test Resource")).toBeInTheDocument();
    expect(getByText("ID")).toBeInTheDocument();
    expect(getByText("123")).toBeInTheDocument();
  });

  it("disables the delete button when requiredConfirmation is not in the input (empty state)", async () => {
    const { getByRole } = renderWithProviders(<DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />);
    const deleteButton = getByRole("button", { name: /delete/i });
    expect(deleteButton).toBeDisabled();
  });

  it("enables the delete button when the input equals requiredConfirmation", async () => {
    const user = userEvent.setup();
    const { getByRole, getByPlaceholderText } = renderWithProviders(
      <DeleteResourceModal {...defaultProps} requiredConfirmation="DELETE" />,
    );
    const input = getByPlaceholderText("DELETE");
    await user.type(input, "DELETE");
    const deleteButton = getByRole("button", { name: /delete/i });
    expect(deleteButton).not.toBeDisabled();
  });
});
