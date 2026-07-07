import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import Sidebar from "./leftnav";

vi.mock("../utils/roles", () => {
  return {
    all_admin_roles: ["admin", "admin_viewer"],
    internalUserRoles: ["internal"],
    rolesWithWriteAccess: ["admin", "internal"],
    rolesAllowedToViewWriteScopedPages: ["admin", "internal", "admin_viewer"],
    isAdminRole: (role: string) => role === "admin" || role === "admin_viewer",
    isUserTeamAdminForAnyTeam: () => false,
  };
});

const { mockUseAuthorized, mockUseOrganizations } = vi.hoisted(() => {
  const mockUseAuthorized = vi.fn(() => ({
    userId: "test-user-id",
    accessToken: "test-access-token",
    userRole: "admin",
    token: "test-token",
    userEmail: "test@example.com",
    premiumUser: false,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  }));

  const mockUseOrganizations = vi.fn(() => ({
    data: [],
    isLoading: false,
    error: null,
  }));

  return { mockUseAuthorized, mockUseOrganizations };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: mockUseOrganizations,
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [], isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => {
  return {
    useUIConfig: () => ({
      data: { admin_ui_disabled: false },
      isLoading: false,
    }),
  };
});

describe("Sidebar (leftnav)", () => {
  const defaultProps = {
    setPage: vi.fn(),
    defaultSelectedKey: "api-keys",
    collapsed: false,
  };

  it("renders all top-level (non-nested) tabs for admin", () => {
    renderWithProviders(<Sidebar {...defaultProps} />);

    const topLevelLabels = [
      "Virtual Keys",
      "Playground",
      "Models + Endpoints",
      "Agentic",
      "MCP Servers",
      "Guardrails",
      "Policies",
      "Tools",
      "Usage",
      "Logs",
      "Guardrails Monitor",
      "Teams",
      "Internal Users",
      "Organizations",
      "Access Groups",
      "Budgets",
      "API Reference",
      "AI Hub",
      "Learning Resources",
      "Experimental",
      "Settings",
    ];

    topLevelLabels.forEach((label) => {
      expect(screen.getByText(label)).toBeInTheDocument();
    });
  });

  it("hides Chat by default", () => {
    renderWithProviders(<Sidebar {...defaultProps} />);
    expect(screen.queryByText("Chat")).not.toBeInTheDocument();
  });

  it("shows Chat when enableChatUI is true", () => {
    renderWithProviders(<Sidebar {...defaultProps} enableChatUI />);
    expect(screen.getByText("Chat")).toBeInTheDocument();
  });

  it("expands a nested tab to reveal its children (Tools > Search Tools)", async () => {
    renderWithProviders(<Sidebar {...defaultProps} />);

    expect(screen.queryByText("Search Tools")).not.toBeInTheDocument();
    act(() => {
      fireEvent.click(screen.getByText("Tools"));
    });
    await waitFor(() => {
      expect(screen.getByText("Search Tools")).toBeInTheDocument();
    });
  });

  describe("Admin Viewer parity", () => {
    const adminViewerAuth = {
      userId: "admin-viewer-user-id",
      accessToken: "test-access-token",
      userRole: "admin_viewer",
      token: "test-token",
      userEmail: "viewer@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    };

    it("hides Playground from Admin Viewer (cost-incurring action)", () => {
      mockUseAuthorized.mockReturnValueOnce(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.queryByText("Playground")).not.toBeInTheDocument();
    });

    it("shows Models + Endpoints to Admin Viewer (read-only)", () => {
      mockUseAuthorized.mockReturnValueOnce(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.getByText("Models + Endpoints")).toBeInTheDocument();
    });

    it("shows Agents (under Agentic) to Admin Viewer (read-only)", async () => {
      mockUseAuthorized.mockReturnValueOnce(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      act(() => {
        fireEvent.click(screen.getByText("Agentic"));
      });
      await waitFor(() => {
        expect(screen.getByText("Agents")).toBeInTheDocument();
      });
    });

    it("shows Logs to Admin Viewer", () => {
      mockUseAuthorized.mockReturnValueOnce(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.getByText("Logs")).toBeInTheDocument();
    });
  });

  it("should show Organizations tab for organization admins", () => {
    mockUseAuthorized.mockReturnValueOnce({
      userId: "org-admin-user-id",
      accessToken: "test-access-token",
      userRole: "viewer",
      token: "test-token",
      userEmail: "orgadmin@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

    mockUseOrganizations.mockReturnValueOnce({
      data: [
        {
          organization_id: "org-1",
          organization_name: "Test Organization",
          spend: 0,
          max_budget: null,
          models: [],
          tpm_limit: null,
          rpm_limit: null,
          members: [
            {
              user_id: "org-admin-user-id",
              user_role: "org_admin",
            },
          ],
        },
      ],
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Sidebar {...defaultProps} />);

    expect(screen.getByText("Organizations")).toBeInTheDocument();
  });
});
