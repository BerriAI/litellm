import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelHubTablePage from "./page";

const mockUseAuthorized = vi.fn();
const mockUseTeams = vi.fn();
const hubProps = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => mockUseTeams() }));
vi.mock("@/components/AIHub/ModelHubTable", () => ({
  default: (props: { canAddModels: boolean }) => {
    hubProps(props);
    return <div data-testid="model-hub-table" />;
  },
}));
vi.mock("@/components/public_model_hub", () => ({
  default: () => <div data-testid="public-model-hub" />,
}));

const session = (userRole: string, isViewOnly = false) => ({
  accessToken: "token",
  userRole,
  userId: "user-1",
  premiumUser: false,
  isViewOnly,
});

const teamAdminOf = (userId: string) => [
  { team_id: "team-1", members_with_roles: [{ user_id: userId, role: "admin" }] },
];

describe("ModelHubTablePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseTeams.mockReturnValue({ data: [] });
  });

  it("lets a proxy admin add models from the garden", () => {
    mockUseAuthorized.mockReturnValue(session("Admin"));
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: true }));
  });

  it("keeps the garden read-only for a view-only admin", () => {
    mockUseAuthorized.mockReturnValue(session("Admin", true));
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: false }));
  });

  it("lets an org admin who administers a team add models, the way the Models page does", () => {
    mockUseAuthorized.mockReturnValue(session("org_admin"));
    mockUseTeams.mockReturnValue({ data: teamAdminOf("user-1") });
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: true }));
  });

  it("keeps the garden read-only for an org admin with no team to add models to", () => {
    mockUseAuthorized.mockReturnValue(session("org_admin"));
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: false }));
  });

  it("sends a non-admin to the public hub", () => {
    mockUseAuthorized.mockReturnValue(session("internal_user"));
    render(<ModelHubTablePage />);
    expect(screen.getByTestId("public-model-hub")).toBeInTheDocument();
    expect(hubProps).not.toHaveBeenCalled();
  });
});
