import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import UsagePage from "./page";
import UsagePageView from "@/components/UsagePage/components/UsagePageView";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

const { mockUseOrganizations, mockUseAuthorized, mockUseTeams } = vi.hoisted(() => ({
  mockUseOrganizations: vi.fn(),
  mockUseAuthorized: vi.fn(),
  mockUseTeams: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: mockUseOrganizations,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: mockUseTeams,
}));

vi.mock("@/components/UsagePage/components/UsagePageView", () => ({
  default: vi.fn(() => <div data-testid="usage-page-view-stub" />),
}));

describe("Usage dashboard page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({
      accessToken: "token",
      userRole: "org_admin",
      userId: "user-1",
      premiumUser: false,
    });
    mockUseTeams.mockReturnValue({ teams: [{ team_id: "team-1" }] });
    mockUseOrganizations.mockReturnValue({ data: undefined, isLoading: false });
  });

  it("calls useOrganizations and passes fetched organizations to UsagePageView", () => {
    const organizations = [{ organization_id: "org-1", organization_alias: "Acme" }];
    mockUseOrganizations.mockReturnValue({ data: organizations, isLoading: false });

    renderWithProviders(<UsagePage />);

    expect(useOrganizations).toHaveBeenCalled();
    expect(vi.mocked(UsagePageView).mock.calls[0][0]).toEqual({
      teams: [{ team_id: "team-1" }],
      organizations,
    });
  });

  it("passes empty organizations array when useOrganizations returns no data", () => {
    mockUseOrganizations.mockReturnValue({ data: undefined, isLoading: false });

    renderWithProviders(<UsagePage />);

    expect(vi.mocked(UsagePageView).mock.calls[0][0]).toEqual({
      teams: [{ team_id: "team-1" }],
      organizations: [],
    });
  });
});
