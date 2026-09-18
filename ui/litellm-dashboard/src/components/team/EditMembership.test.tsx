import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import EditMembership from "./EditMembership";

describe("EditMembership", () => {
  const mockOnCancel = vi.fn();
  const mockOnSubmit = vi.fn();

  const defaultConfig = {
    title: "Add Member",
    roleOptions: [
      { label: "Admin", value: "admin" },
      { label: "Member", value: "member" },
    ],
    defaultRole: "member",
    showEmail: true,
    showUserId: false,
  };

  it("should render", () => {
    renderWithProviders(
      <EditMembership
        visible={true}
        onCancel={mockOnCancel}
        onSubmit={mockOnSubmit}
        mode="add"
        config={defaultConfig}
      />,
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Role")).toBeInTheDocument();
  });

  it("should submit form data when adding a member", async () => {
    renderWithProviders(
      <EditMembership
        visible={true}
        onCancel={mockOnCancel}
        onSubmit={mockOnSubmit}
        mode="add"
        config={defaultConfig}
      />,
    );

    const emailInput = screen.getByPlaceholderText("user@example.com");
    const submitButton = screen.getByRole("button", { name: "Add Member" });

    act(() => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    });

    await waitFor(() => {
      expect(emailInput).toHaveValue("test@example.com");
    });

    act(() => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          user_email: "test@example.com",
          role: "member",
        }),
      );
    });
  });
});
