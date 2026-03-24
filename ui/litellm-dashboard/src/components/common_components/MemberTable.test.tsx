import { renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";
import MemberTable from "./MemberTable";
import type { Member } from "@/components/networking";

vi.mock(
  "@/components/common_components/IconActionButton/TableIconActionButtons/TableIconActionButton",
  () => ({
    default: ({
      onClick,
      dataTestId,
      variant,
    }: {
      onClick: () => void;
      dataTestId?: string;
      variant: string;
    }) => (
      <button data-testid={dataTestId} onClick={onClick}>
        {variant}
      </button>
    ),
  })
);

const members: Member[] = [
  { user_id: "user-1", user_email: "alice@example.com", role: "admin" },
  { user_id: "user-2", user_email: "bob@example.com", role: "user" },
  {
    user_id: "default_user_id",
    user_email: "proxy@admin.com",
    role: "admin",
  },
];

describe("MemberTable", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("should render", () => {
    renderWithProviders(
      <MemberTable
        members={members}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    expect(screen.getByText("3 Members")).toBeInTheDocument();
  });

  it("should display member emails in the table", () => {
    renderWithProviders(
      <MemberTable
        members={members}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    expect(screen.getByText("alice@example.com")).toBeInTheDocument();
    expect(screen.getByText("bob@example.com")).toBeInTheDocument();
  });

  it("should show Default Proxy Admin tag for default_user_id", () => {
    renderWithProviders(
      <MemberTable
        members={members}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });

  it("should show singular 'Member' label when there is only one member", () => {
    renderWithProviders(
      <MemberTable
        members={[members[0]]}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    expect(screen.getByText("1 Member")).toBeInTheDocument();
  });

  it("should call onEdit when the edit button is clicked", async () => {
    const onEdit = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <MemberTable
        members={[members[0]]}
        canEdit={true}
        onEdit={onEdit}
        onDelete={vi.fn()}
      />
    );

    await user.click(screen.getByTestId("edit-member"));
    expect(onEdit).toHaveBeenCalledWith(members[0]);
  });

  it("should call onDelete when the delete button is clicked", async () => {
    const onDelete = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <MemberTable
        members={[members[0]]}
        canEdit={true}
        onEdit={vi.fn()}
        onDelete={onDelete}
      />
    );

    await user.click(screen.getByTestId("delete-member"));
    expect(onDelete).toHaveBeenCalledWith(members[0]);
  });

  it("should not show action buttons when canEdit is false", () => {
    renderWithProviders(
      <MemberTable
        members={[members[0]]}
        canEdit={false}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    expect(screen.queryByTestId("edit-member")).not.toBeInTheDocument();
    expect(screen.queryByTestId("delete-member")).not.toBeInTheDocument();
  });

  it("should show the Add Member button when onAddMember is provided and canEdit is true", () => {
    renderWithProviders(
      <MemberTable
        members={members}
        canEdit={true}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onAddMember={vi.fn()}
      />
    );
    expect(
      screen.getByRole("button", { name: /add member/i })
    ).toBeInTheDocument();
  });

  it("should call onAddMember when the Add Member button is clicked", async () => {
    const onAddMember = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <MemberTable
        members={members}
        canEdit={true}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onAddMember={onAddMember}
      />
    );

    await user.click(screen.getByRole("button", { name: /add member/i }));
    expect(onAddMember).toHaveBeenCalled();
  });

  it("should hide delete for a member when showDeleteForMember returns false", () => {
    renderWithProviders(
      <MemberTable
        members={[members[0]]}
        canEdit={true}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        showDeleteForMember={() => false}
      />
    );
    expect(screen.queryByTestId("delete-member")).not.toBeInTheDocument();
    expect(screen.getByTestId("edit-member")).toBeInTheDocument();
  });
});
