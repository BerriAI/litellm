import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EditUserModal from "./edit_user";

const possibleUIRoles: Record<string, Record<string, string>> = {
  admin: { ui_label: "Admin", description: "Full access" },
  user: { ui_label: "User", description: "Standard access" },
};

const mockUser = {
  user_id: "user-123",
  user_email: "test@example.com",
  user_role: "user",
  spend: 10.5,
  max_budget: 100,
};

describe("EditUserModal", () => {
  const defaultProps = {
    visible: true,
    possibleUIRoles,
    onCancel: vi.fn(),
    user: mockUser,
    onSubmit: vi.fn(),
  };

  it("should render", () => {
    render(<EditUserModal {...defaultProps} />);
    expect(screen.getByText(/edit user user-123/i)).toBeInTheDocument();
  });

  it("should return null when user is null", () => {
    render(<EditUserModal {...defaultProps} user={null} />);
    expect(screen.queryByText(/edit user/i)).not.toBeInTheDocument();
  });

  it("should display the user email field", () => {
    render(<EditUserModal {...defaultProps} />);
    expect(screen.getByText("User Email")).toBeInTheDocument();
  });

  it("should display the user role field", () => {
    render(<EditUserModal {...defaultProps} />);
    expect(screen.getByText("User Role")).toBeInTheDocument();
  });

  it("should call onCancel when cancel is triggered", async () => {
    const onCancel = vi.fn();
    const user = userEvent.setup();
    render(<EditUserModal {...defaultProps} onCancel={onCancel} />);

    // Click the X close button on the modal
    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(onCancel).toHaveBeenCalled();
  });
});
