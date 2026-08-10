import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import Sidebar, { menuGroups, getBreadcrumb } from "./leftnav";
import { resources } from "@/i18n/resources";

const { mockLanguageState } = vi.hoisted(() => ({ mockLanguageState: { current: "en" as "en" | "ru" } }));

vi.mock("@/i18n/I18nProvider", () => ({
  useDashboardLanguage: () => ({ language: mockLanguageState.current, setLanguage: vi.fn() }),
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
vi.mock("@/contexts/ThemeContext", () => ({
  useTheme: () => ({ logoUrl: null, faviconUrl: null, setLogoUrl: vi.fn(), setFaviconUrl: vi.fn() }),
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
    mockLanguageState.current = "en";
  });

  it("renders the complete visible sidebar in Russian without localizing breadcrumbs", async () => {
    mockLanguageState.current = "ru";
    renderWithProviders(<Sidebar {...defaultProps} />);

    [
      "AI-ШЛЮЗ",
      "Виртуальные ключи",
      "Песочница",
      "Модели и эндпоинты",
      "Агентные функции",
      "MCP-серверы",
      "Навыки",
      "Ограничители",
      "Политики",
      "Инструменты",
      "НАБЛЮДАЕМОСТЬ",
      "Использование",
      "Оптимизация затрат",
      "Логи",
      "Мониторинг ограничителей",
      "УПРАВЛЕНИЕ ДОСТУПОМ",
      "Команды",
      "Внутренние пользователи",
      "Организации",
      "Группы доступа",
      "Бюджеты",
      "ИНСТРУМЕНТЫ РАЗРАБОТЧИКА",
      "Справочник API",
      "Каталог AI",
      "Учебные материалы",
      "Кэш ответов",
      "Экспериментальные",
      "НАСТРОЙКИ",
    ].forEach((label) => expect(screen.getByText(label)).toBeInTheDocument());

    fireEvent.click(screen.getByText("Инструменты"));
    await waitFor(() => expect(screen.getByText("Поиск инструментов")).toBeInTheDocument());
    expect(screen.getByText("Векторные хранилища")).toBeInTheDocument();
    expect(screen.getByText("Политики инструментов")).toBeInTheDocument();

    expect(screen.getByText("Бета")).toBeInTheDocument();
    expect(screen.getByText("Новое")).toBeInTheDocument();
    expect(getBreadcrumb("api-keys")).toEqual({ section: "AI Gateway", title: "Virtual Keys" });
  });

  it("has English and Russian translations for every navigation key", () => {
    const translations = resources as {
      en: { translation: { sidebar: { groups: Record<string, string>; items: Record<string, string> } } };
      ru: { translation: { sidebar: { groups: Record<string, string>; items: Record<string, string> } } };
    };

    menuGroups.forEach((group) => {
      expect(translations.en.translation.sidebar.groups[group.groupLabel]).toBeTruthy();
      expect(translations.ru.translation.sidebar.groups[group.groupLabel]).toBeTruthy();
      group.items.forEach((item) => {
        expect(translations.en.translation.sidebar.items[item.key]).toBeTruthy();
        expect(translations.ru.translation.sidebar.items[item.key]).toBeTruthy();
        item.children?.forEach((child) => {
          expect(translations.en.translation.sidebar.items[child.key]).toBeTruthy();
          expect(translations.ru.translation.sidebar.items[child.key]).toBeTruthy();
        });
      });
    });
  });

  it("should link the logo to the UI home route rather than the proxy origin", () => {
    renderWithProviders(<Sidebar {...defaultProps} />);

    expect(screen.getByRole("link", { name: /litellm home/i })).toHaveAttribute("href", "/ui");
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

  describe("capability-gated Tools children", () => {
    const internalAuth = {
      userId: "internal-user-id",
      accessToken: "test-access-token",
      userRole: "internal",
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
    expect(costOptimization!.textContent).toContain("Cost Optimization");
    expect(costOptimization!.textContent).toContain("Beta");

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
