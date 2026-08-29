import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import Sidebar, { menuGroups, getBreadcrumb } from "./leftnav";

vi.mock("../utils/roles", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../utils/roles")>();
  return {
    ...actual,
    all_admin_roles: ["admin", "admin_viewer"],
    old_admin_roles: ["admin", "admin_viewer"],
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
    isViewOnly: false,
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

// The redesigned sidebar reads the custom logo from ThemeContext; the test tree
// has no ThemeProvider, so stub the hook.
const unbrandedTheme = () => ({
  logoUrl: null as string | null,
  logoUrlDark: null as string | null,
  faviconUrl: null as string | null,
  setLogoUrl: vi.fn(),
  setLogoUrlDark: vi.fn(),
  setFaviconUrl: vi.fn(),
});
let mockUseThemeImpl = unbrandedTheme;
vi.mock("@/contexts/ThemeContext", () => ({
  useTheme: () => mockUseThemeImpl(),
}));

// Version tag + logout target come from network hooks; keep them inert in unit tests.
vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: () => ({ data: undefined }),
}));
vi.mock("@/app/(dashboard)/hooks/useLogout", () => ({
  useLogout: () => vi.fn(),
}));

const collectNavKeys = (): string[] =>
  menuGroups.flatMap((group) => group.items.flatMap((item) => [item.key, ...(item.children ?? []).map((c) => c.key)]));

// Every place a page id appears in the nav, as "GROUP" for a top-level item or
// "GROUP > parentKey" for a child.
const placementsOf = (page: string): string[] =>
  menuGroups.flatMap((group) => [
    ...group.items.filter((item) => item.page === page).map(() => group.groupLabel),
    ...group.items.flatMap((item) =>
      (item.children ?? []).filter((child) => child.page === page).map(() => `${group.groupLabel} > ${item.key}`),
    ),
  ]);

describe("Sidebar (leftnav)", () => {
  const defaultProps = {
    setPage: vi.fn(),
    defaultSelectedKey: "api-keys",
    collapsed: false,
  };

  afterEach(() => {
    mockUseAuthorized.mockReset();
    mockUseOrganizations.mockReset();
    mockUseThemeImpl = unbrandedTheme;
  });

  it("should link the logo to the UI home route rather than the proxy origin", () => {
    renderWithProviders(<Sidebar {...defaultProps} />);

    expect(screen.getByRole("link", { name: /litellm home/i })).toHaveAttribute("href", "/ui");
  });

  it("pairs the logo with a dark-mode variant that swaps on the dark class", () => {
    renderWithProviders(<Sidebar {...defaultProps} />);

    const [light, dark] = Array.from(screen.getByRole("link", { name: /litellm home/i }).querySelectorAll("img"));
    const classesOf = (el: Element) => new Set(el.className.split(/\s+/));

    const lightSrc = light.getAttribute("src") ?? "";
    expect(light).toHaveAttribute("src", expect.stringMatching(/\/get_image$/));
    expect(dark).toHaveAttribute("src", `${lightSrc}?theme=dark`);
    expect(classesOf(light).has("dark:hidden")).toBe(true);
    expect(classesOf(light).has("hidden")).toBe(false);
    expect(classesOf(dark).has("hidden")).toBe(true);
    expect(classesOf(dark).has("dark:block")).toBe(true);
  });

  it("prefers a configured dark logo over the light one in dark mode", () => {
    mockUseThemeImpl = () => ({
      ...unbrandedTheme(),
      logoUrl: "https://cdn.example.com/logo.png",
      logoUrlDark: "https://cdn.example.com/logo-dark.png",
    });
    renderWithProviders(<Sidebar {...defaultProps} />);

    const [light, dark] = Array.from(screen.getByRole("link", { name: /litellm home/i }).querySelectorAll("img"));

    expect(light).toHaveAttribute("src", "https://cdn.example.com/logo.png");
    expect(dark).toHaveAttribute("src", "https://cdn.example.com/logo-dark.png");
  });

  it("reuses the light custom logo in dark mode when no dark one is configured", () => {
    mockUseThemeImpl = () => ({ ...unbrandedTheme(), logoUrl: "https://cdn.example.com/logo.png" });
    renderWithProviders(<Sidebar {...defaultProps} />);

    const [light, dark] = Array.from(screen.getByRole("link", { name: /litellm home/i }).querySelectorAll("img"));

    expect(light).toHaveAttribute("src", "https://cdn.example.com/logo.png");
    expect(dark).toHaveAttribute("src", "https://cdn.example.com/logo.png");
  });

  it("falls back to the light logo when a configured dark logo fails to load", () => {
    mockUseThemeImpl = () => ({
      ...unbrandedTheme(),
      logoUrl: "https://cdn.example.com/logo.png",
      logoUrlDark: "https://cdn.example.com/gone.png",
    });
    renderWithProviders(<Sidebar {...defaultProps} />);

    const [, dark] = Array.from(screen.getByRole("link", { name: /litellm home/i }).querySelectorAll("img"));
    expect(dark).toHaveAttribute("src", "https://cdn.example.com/gone.png");

    fireEvent.error(dark);

    expect(dark).toHaveAttribute("src", "https://cdn.example.com/logo.png");
  });

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
  it("keeps Router Settings as a single Settings child", () => {
    // Router Settings is admin-only, so getAvailablePages() filters it out entirely and the
    // page_utils duplicate-key guard cannot see it. Walk menuGroups directly, otherwise a
    // stray duplicate placement ships silently.
    expect(placementsOf("router-settings")).toEqual(["SETTINGS > settings"]);
  });

  it("has no duplicate keys among all menu items and their children", () => {
    // React keys must be unique across the whole nav config, otherwise the
    // active-item highlight and group expansion collide.
    const keys = collectNavKeys();
    const duplicates = keys.filter((key, i) => keys.indexOf(key) !== i);
    expect(duplicates).toEqual([]);
  });

  describe("Admin Viewer parity", () => {
    // Admin Viewer follows a "read parity with Proxy Admin, no writes, no
    // cost-incurring actions" rule. The session hook presents the viewer as
    // an admin (`userRole: "admin"`) with `isViewOnly: true`; Playground
    // stays hidden (incurs LLM cost) via the isViewOnly flag, while every
    // admin page (Models + Endpoints, Agents, Logs, ...) is visible read-only.
    const adminViewerAuth = {
      userId: "admin-viewer-user-id",
      accessToken: "test-access-token",
      userRole: "admin",
      isViewOnly: true,
      token: "test-token",
      userEmail: "viewer@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    };

    it("hides Playground from Admin Viewer (cost-incurring action)", () => {
      mockUseAuthorized.mockReturnValue(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.queryByText("Playground")).not.toBeInTheDocument();
    });

    it("shows Models + Endpoints to Admin Viewer (read-only)", () => {
      mockUseAuthorized.mockReturnValue(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.getByText("Models + Endpoints")).toBeInTheDocument();
    });

    it("shows Agents (under Agentic) to Admin Viewer (read-only)", async () => {
      mockUseAuthorized.mockReturnValue(adminViewerAuth);
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
      mockUseAuthorized.mockReturnValue(adminViewerAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);
      expect(screen.getByText("Logs")).toBeInTheDocument();
    });
  });

  describe("capability-gated Tools children", () => {
    const internalAuth = {
      userId: "internal-user-id",
      accessToken: "test-access-token",
      userRole: "internal",
      isViewOnly: false,
      token: "test-token",
      userEmail: "internal@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    };

    afterEach(() => {
      mockUseAuthorized.mockReset();
    });

    it("should hide Tool Policies from internal users while keeping other Tools children", async () => {
      mockUseAuthorized.mockReturnValue(internalAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Tools"));
      });
      await waitFor(() => {
        expect(screen.getByText("Search Tools")).toBeInTheDocument();
      });
      expect(screen.queryByText("Tool Policies")).not.toBeInTheDocument();
    });

    it("should show Tool Policies to admins", async () => {
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Tools"));
      });
      await waitFor(() => {
        expect(screen.getByText("Tool Policies")).toBeInTheDocument();
      });
    });

    it("should hide the Policies entry from internal users while keeping Guardrails", () => {
      mockUseAuthorized.mockReturnValue(internalAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);

      expect(screen.getByText("Guardrails")).toBeInTheDocument();
      expect(screen.queryByText("Policies")).not.toBeInTheDocument();
    });

    it("should hide the Prompts entry from internal users while keeping other Experimental children", async () => {
      mockUseAuthorized.mockReturnValue(internalAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Experimental"));
      });
      await waitFor(() => {
        expect(screen.getByText("API Playground")).toBeInTheDocument();
      });
      expect(screen.queryByText("Prompts")).not.toBeInTheDocument();
    });

    it("should hide Old Usage from internal users while keeping other Experimental children", async () => {
      mockUseAuthorized.mockReturnValue(internalAuth);
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Experimental"));
      });
      await waitFor(() => {
        expect(screen.getByText("API Playground")).toBeInTheDocument();
      });
      expect(screen.queryByText("Old Usage")).not.toBeInTheDocument();
    });

    it("should show Old Usage to admins", async () => {
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Experimental"));
      });
      await waitFor(() => {
        expect(screen.getByText("Old Usage")).toBeInTheDocument();
      });
    });
  });

  // Workflow Runs, Memory and Guardrails Monitor render a shell and then 401
  // for every non-proxy-admin role, because their page-load routes sit outside
  // internal_user_routes / self_managed_routes. Cost Optimization does not:
  // its primary call is /user/daily/activity, which every role may make, so
  // the entry stays and only its proxy-wide tabs are gated inside the page.
  describe("capability-gated pages whose data is proxy-admin-only", () => {
    const authFor = (userRole: string) => ({
      userId: "some-user-id",
      accessToken: "test-access-token",
      userRole,
      isViewOnly: false,
      token: "test-token",
      userEmail: "someone@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

    afterEach(() => {
      mockUseAuthorized.mockReset();
    });

    it("hides Workflow Runs and Memory from an internal user under Agentic", async () => {
      mockUseAuthorized.mockReturnValue(authFor("internal"));
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Agentic"));
      });
      // Liveness gate: the sibling Agents child stays visible to this role, so
      // the absences below mean the gate fired, not that the group never opened.
      await waitFor(() => {
        expect(screen.getByText("Agents")).toBeInTheDocument();
      });
      expect(screen.queryByText("Workflow Runs")).not.toBeInTheDocument();
      expect(screen.queryByText("Memory")).not.toBeInTheDocument();
    });

    // An org admin's session role is "Org Admin", which no capability list
    // carries, and the proxy denies these routes to org admins too because
    // `_user_is_org_admin` needs an organization_id the page-load GET never sends.
    // Agents is already out of reach for this role, so gating the other two
    // empties the Agentic group entirely and the parent must go with it rather
    // than degrade into a leaf link to the non-route `?page=agentic`.
    it("drops the whole Agentic group for an org admin once its last child is gated", () => {
      mockUseAuthorized.mockReturnValue(authFor("org_admin"));
      renderWithProviders(<Sidebar {...defaultProps} />);

      // Liveness gate: Logs carries no role list, so it proves the sidebar rendered.
      expect(screen.getByText("Logs")).toBeInTheDocument();
      expect(screen.queryByText("Agentic")).not.toBeInTheDocument();
      expect(screen.queryByText("Workflow Runs")).not.toBeInTheDocument();
      expect(screen.queryByText("Memory")).not.toBeInTheDocument();
    });

    it("keeps the Agentic group for an internal user, who can still see Agents", () => {
      mockUseAuthorized.mockReturnValue(authFor("internal"));
      renderWithProviders(<Sidebar {...defaultProps} />);

      expect(screen.getByText("Agentic")).toBeInTheDocument();
    });

    it("shows Workflow Runs and Memory to admins", async () => {
      renderWithProviders(<Sidebar {...defaultProps} />);

      act(() => {
        fireEvent.click(screen.getByText("Agentic"));
      });
      await waitFor(() => {
        expect(screen.getByText("Workflow Runs")).toBeInTheDocument();
      });
      expect(screen.getByText("Memory")).toBeInTheDocument();
    });

    it("hides Guardrails Monitor from an internal user while keeping Usage and Cost Optimization", () => {
      mockUseAuthorized.mockReturnValue(authFor("internal"));
      renderWithProviders(<Sidebar {...defaultProps} />);

      expect(screen.queryByText("Guardrails Monitor")).not.toBeInTheDocument();
      expect(screen.getByText("Usage")).toBeInTheDocument();
      expect(screen.getByText("Cost Optimization")).toBeInTheDocument();
    });

    it("shows Guardrails Monitor to admins", () => {
      renderWithProviders(<Sidebar {...defaultProps} />);

      expect(screen.getByText("Guardrails Monitor")).toBeInTheDocument();
    });
  });

  it("should show Organizations tab for organization admins", () => {
    mockUseAuthorized.mockReturnValue({
      userId: "org-admin-user-id",
      accessToken: "test-access-token",
      userRole: "viewer",
      isViewOnly: false,
      token: "test-token",
      userEmail: "orgadmin@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

    mockUseOrganizations.mockReturnValue({
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

  it("marks the selected page's nav item active", () => {
    renderWithProviders(<Sidebar {...defaultProps} defaultSelectedKey="logs" />);
    const logs = screen.getByText("Logs").closest("a");
    expect(logs).toHaveAttribute("data-active", "true");
    // A different item must not be active.
    expect(screen.getByText("Virtual Keys").closest("a")).not.toHaveAttribute("data-active");
  });

  it("hides labels but keeps items reachable (icon + link) when collapsed to the rail", () => {
    const { container } = renderWithProviders(<Sidebar {...defaultProps} collapsed />);
    expect(container.querySelector('[data-slot="sidebar"]')).toHaveAttribute("data-collapsed", "true");
    // The item stays navigable in the icon-only rail: its link still renders with
    // an icon (asserting the <a> + svg, not the text, so a removed icon would
    // fail here), while the label is present but CSS-hidden.
    const label = screen.getByText("Virtual Keys");
    const link = label.closest("a");
    expect(link).not.toBeNull();
    expect(link!.querySelector("svg")).not.toBeNull();
    expect(label).toHaveClass("group-data-[collapsed=true]/sidebar:hidden");
  });

  it("shows Cost Optimization with a Beta badge and no feature-flag gate", () => {
    const { container } = renderWithProviders(<Sidebar {...defaultProps} enableProjectsUI={false} />);

    const costOptimization = container.querySelector('a[href*="cost-optimization"]');
    expect(costOptimization).not.toBeNull();
    expect(costOptimization!).toHaveTextContent(/Cost Optimization/);
    expect(costOptimization!).toHaveTextContent(/Beta/);

    expect(container.querySelector('a[href*="projects"]')).toBeNull();
  });

  it("keeps a readable collapsed-rail tooltip for items whose label carries a badge", () => {
    const { container } = renderWithProviders(<Sidebar {...defaultProps} enableProjectsUI collapsed />);

    expect(container.querySelector('a[href*="cost-optimization"]')).toHaveAttribute("title", "Cost Optimization");
    expect(container.querySelector('a[href*="projects"]')).toHaveAttribute("title", "Projects");
  });
});

describe("getBreadcrumb", () => {
  it("resolves a top-level page to its section + title", () => {
    expect(getBreadcrumb("api-keys")).toEqual({ section: "AI Gateway", title: "Virtual Keys" });
    expect(getBreadcrumb("logs")).toEqual({ section: "Observability", title: "Logs" });
  });

  it("resolves a nested child page to its parent section", () => {
    expect(getBreadcrumb("search-tools")).toEqual({ section: "AI Gateway", title: "Search Tools" });
  });

  it("resolves router-settings under the Settings section", () => {
    expect(getBreadcrumb("router-settings")).toEqual({ section: "Settings", title: "Router Settings" });
  });

  it("falls back to a prettified title with no section for unknown pages", () => {
    expect(getBreadcrumb("some-unknown-page")).toEqual({ section: null, title: "Some Unknown Page" });
  });
});
