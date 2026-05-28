import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import Sidebar from "./leftnav";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => "/ui/",
  useSearchParams: () => new URLSearchParams(),
}));

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
  it("has no duplicate keys among all menu items and their children", () => {
    // Helper to recursively extract all keys from Ant Design Menu items
    function getAllKeysFromMenu(wrapper: HTMLElement): string[] {
      const allKeys: string[] = [];
      // Ant Design renders key as data-menu-id or inside attributes, but for this case, we look for text as fallback.
      // For a generic check, here we fetch ids from rendered list items, and also descend into submenus
      const items = wrapper.querySelectorAll("[data-menu-id]");
      items.forEach((item) => {
        const dataMenuId = item.getAttribute("data-menu-id");
        if (dataMenuId) {
          allKeys.push(dataMenuId);
        }
      });
      return allKeys;
    }

    const { container } = renderWithProviders(<Sidebar {...defaultProps} />);
    const allRenderedKeys = getAllKeysFromMenu(container);

    const keySet = new Set<string>();
    const duplicates: string[] = [];
    for (const key of allRenderedKeys) {
      if (keySet.has(key)) {
        duplicates.push(key);
      }
      keySet.add(key);
    }
    expect(duplicates).toHaveLength(0);
  });

  describe("Admin Viewer parity", () => {
    // Admin Viewer follows a "read parity with Proxy Admin, no writes, no
    // cost-incurring actions" rule. Playground stays hidden (incurs LLM
    // cost); Models + Endpoints and Agents must be visible read-only.
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
      // Agents is now nested under the "Agentic" submenu — expand parent
      // first to render the children, then assert Agents is visible.
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

// -------------------------------------------------------------------------
// LIT-3063 — Sidebar nav broken in v1.83.14
//
// Two regressions covered by the helpers introduced in leftnav.tsx:
//   1. The API Reference sidebar entry was rendering with href
//      "/api-reference" instead of "/ui/api-reference", so Ctrl/Cmd+click
//      (or double-click) navigated outside the UI mount and 404ed.
//   2. The legacy (query-param) entries rendered with relative
//      hrefs like "?page=logs" — so on /ui/api-reference the browser
//      resolved them to "/ui/api-reference?page=logs" instead of
//      "/ui/?page=logs", leaving the previous route segment in the URL.
//
// The hrefs are now anchored to the UI root ("<serverRootPath>/ui/")
// regardless of NEXT_PUBLIC_BASE_URL, matching the proxys static
// mount in litellm/proxy/proxy_server.py.
// -------------------------------------------------------------------------
describe("Sidebar (leftnav) — LIT-3063 absolute /ui/ href anchoring", () => {
  const defaultProps = {
    setPage: vi.fn(),
    defaultSelectedKey: "api-keys",
    collapsed: false,
  };

  it("renders the API Reference link with an absolute /ui/api-reference href", () => {
    const { container } = renderWithProviders(<Sidebar {...defaultProps} />);
    const apiRef = Array.from(container.querySelectorAll("a")).find(
      (a) => a.textContent?.trim() === "API Reference",
    );
    expect(apiRef).toBeTruthy();
    // The raw attribute (what Ctrl/Cmd+click follows) must already be /ui/...
    expect(apiRef!.getAttribute("href")).toBe("/ui/api-reference");
  });

  it("renders every legacy (query-param) link anchored to /ui/", () => {
    const { container } = renderWithProviders(<Sidebar {...defaultProps} />);
    const anchors = Array.from(container.querySelectorAll("a"));
    const cases: Record<string, string> = {
      Logs: "page=logs",
      Teams: "page=teams",
      "Virtual Keys": "page=api-keys",
    };
    for (const [label, expected] of Object.entries(cases)) {
      const a = anchors.find((x) => x.textContent?.trim() === label);
      expect(a, `missing link for ${label}`).toBeTruthy();
      const href = a!.getAttribute("href") ?? "";
      // Anchored to /ui/, not relative ?page=...
      expect(href.startsWith("/ui/?"), `${label} href \`${href}\` must start with /ui/?`).toBe(true);
      expect(href).toContain(expected);
    }
  });
});
