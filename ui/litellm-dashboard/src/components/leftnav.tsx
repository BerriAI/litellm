import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  ApiOutlined,
  ApartmentOutlined,
  AppstoreOutlined,
  AuditOutlined,
  BankOutlined,
  BarChartOutlined,
  BgColorsOutlined,
  BlockOutlined,
  BookOutlined,
  CreditCardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  ExportOutlined,
  FileTextOutlined,
  FolderOutlined,
  KeyOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  RobotOutlined,
  SafetyOutlined,
  SearchOutlined,
  SettingOutlined,
  TagsOutlined,
  TeamOutlined,
  ToolOutlined,
  UserOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";
import { ConfigProvider, Layout, Menu } from "antd";
import type { TFunction } from "i18next";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  all_admin_roles,
  internalUserRoles,
  isAdminRole,
  isUserTeamAdminForAnyTeam,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
} from "../utils/roles";
import NewBadge from "./common_components/NewBadge";
import type { Organization } from "./networking";
import UsageIndicator from "./UsageIndicator";
import { MIGRATED_PAGES, migratedHref, legacyPageHref } from "@/utils/migratedPages";
const { Sider } = Layout;

// Define the props type
interface SidebarProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  collapsed?: boolean;
  enabledPagesInternalUsers?: string[] | null;
  enableProjectsUI?: boolean;
  disableAgentsForInternalUsers?: boolean;
  allowAgentsForTeamAdmins?: boolean;
  disableVectorStoresForInternalUsers?: boolean;
  allowVectorStoresForTeamAdmins?: boolean;
}

// Menu item configuration
interface MenuItem {
  key: string;
  page: string;
  label: string | React.ReactNode;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
  external_url?: string;
}

// Group configuration
interface MenuGroup {
  groupLabel: string;
  items: MenuItem[];
  roles?: string[];
}

// Menu groups organized by category; built through a factory so labels resolve
// against the active language
const getMenuGroups = (t: TFunction): MenuGroup[] => [
  {
    groupLabel: t("nav.groups.aiGateway"),
    items: [
      {
        key: "api-keys",
        page: "api-keys",
        label: t("nav.virtualKeys"),
        icon: <KeyOutlined />,
      },
      {
        key: "llm-playground",
        page: "llm-playground",
        label: t("nav.playground"),
        icon: <PlayCircleOutlined />,
        roles: rolesWithWriteAccess,
      },
      {
        key: "models",
        page: "models",
        label: t("nav.modelsEndpoints"),
        icon: <BlockOutlined />,
        // Admin Viewer can view models read-only (write actions are
        // hidden inside the page); Playground above stays write-only.
        roles: rolesAllowedToViewWriteScopedPages,
      },
      {
        key: "agentic",
        page: "agentic",
        label: t("nav.agentic"),
        icon: <RobotOutlined />,
        children: [
          {
            key: "agents",
            page: "agents",
            label: t("nav.agents"),
            icon: <RobotOutlined />,
            // Admin Viewer can view agents read-only (write actions are
            // hidden inside the page); Playground above stays write-only.
            roles: rolesAllowedToViewWriteScopedPages,
          },
          {
            key: "workflows",
            page: "workflows",
            label: t("nav.workflowRuns"),
            icon: <ApartmentOutlined />,
          },
          {
            key: "memory",
            page: "memory",
            label: t("nav.memory"),
            icon: <BookOutlined />,
          },
        ],
      },
      {
        key: "mcp-servers",
        page: "mcp-servers",
        label: t("nav.mcpServers"),
        icon: <ToolOutlined />,
      },
      {
        key: "skills",
        page: "skills",
        label: t("nav.skills"),
        icon: <ApiOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "guardrails",
        page: "guardrails",
        label: t("nav.guardrails"),
        icon: <SafetyOutlined />,
      },
      {
        key: "policies",
        page: "policies",
        label: <span className="flex items-center gap-4">{t("nav.policies")}</span>,
        icon: <AuditOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "tools",
        page: "tools",
        label: t("nav.tools"),
        icon: <ToolOutlined />,
        children: [
          {
            key: "search-tools",
            page: "search-tools",
            label: t("nav.searchTools"),
            icon: <SearchOutlined />,
          },
          {
            key: "vector-stores",
            page: "vector-stores",
            label: t("nav.vectorStores"),
            icon: <DatabaseOutlined />,
          },
          {
            key: "tool-policies",
            page: "tool-policies",
            label: t("nav.toolPolicies"),
            icon: <SafetyOutlined />,
          },
        ],
      },
    ],
  },
  {
    groupLabel: t("nav.groups.observability"),
    items: [
      {
        key: "new_usage",
        page: "new_usage",
        icon: <BarChartOutlined />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: t("nav.usage"),
      },
      {
        key: "logs",
        page: "logs",
        label: t("nav.logs"),
        icon: <LineChartOutlined />,
      },
      {
        key: "guardrails-monitor",
        page: "guardrails-monitor",
        label: t("nav.guardrailsMonitor"),
        icon: <SafetyOutlined />,
        roles: [...all_admin_roles, ...internalUserRoles],
      },
    ],
  },
  {
    groupLabel: t("nav.groups.accessControl"),
    items: [
      {
        key: "teams",
        page: "teams",
        label: t("nav.teams"),
        icon: <TeamOutlined />,
      },
      {
        key: "projects",
        page: "projects",
        label: (
          <span className="flex items-center gap-2">
            {t("nav.projects")} <NewBadge />
          </span>
        ),
        icon: <FolderOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "users",
        page: "users",
        label: t("nav.internalUsers"),
        icon: <UserOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "organizations",
        page: "organizations",
        label: t("nav.organizations"),
        icon: <BankOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "access-groups",
        page: "access-groups",
        label: t("nav.accessGroups"),
        icon: <BlockOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "budgets",
        page: "budgets",
        label: t("nav.budgets"),
        icon: <CreditCardOutlined />,
        roles: all_admin_roles,
      },
    ],
  },
  {
    groupLabel: t("nav.groups.developerTools"),
    items: [
      {
        key: "api_ref",
        page: "api_ref",
        label: t("nav.apiReference"),
        icon: <ApiOutlined />,
      },
      {
        key: "model-hub-table",
        page: "model-hub-table",
        label: t("nav.aiHub"),
        icon: <AppstoreOutlined />,
      },

      {
        key: "learning-resources",
        page: "learning-resources",
        label: t("nav.learningResources"),
        icon: <BookOutlined />,
        external_url: "https://models.litellm.ai/cookbook",
      },
      {
        key: "experimental",
        page: "experimental",
        label: t("nav.experimental"),
        icon: <ExperimentOutlined />,
        children: [
          {
            key: "caching",
            page: "caching",
            label: t("nav.caching"),
            icon: <DatabaseOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "prompts",
            page: "prompts",
            label: t("nav.prompts"),
            icon: <FileTextOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "transform-request",
            page: "transform-request",
            label: t("nav.apiPlayground"),
            icon: <ApiOutlined />,
            roles: [...all_admin_roles, ...internalUserRoles],
          },
          {
            key: "tag-management",
            page: "tag-management",
            label: t("nav.tagManagement"),
            icon: <TagsOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "4",
            page: "usage",
            label: t("nav.oldUsage"),
            icon: <BarChartOutlined />,
          },
        ],
      },
    ],
  },
  {
    groupLabel: t("nav.groups.settings"),
    roles: all_admin_roles,
    items: [
      {
        key: "settings",
        page: "settings",
        label: (
          <span className="flex items-center gap-2">
            {t("nav.settings")} <NewBadge />
          </span>
        ),
        icon: <SettingOutlined />,
        roles: all_admin_roles,
        children: [
          {
            key: "router-settings",
            page: "router-settings",
            label: t("nav.routerSettings"),
            icon: <SettingOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "logging-and-alerts",
            page: "logging-and-alerts",
            label: t("nav.loggingAlerts"),
            icon: <SettingOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "admin-panel",
            page: "admin-panel",
            label: (
              <span className="flex items-center gap-2">
                {t("nav.adminSettings")}{" "}
                <NewBadge dot>
                  <span />
                </NewBadge>
              </span>
            ),
            icon: <SettingOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "cost-tracking",
            page: "cost-tracking",
            label: t("nav.costTracking"),
            icon: <BarChartOutlined />,
            roles: all_admin_roles,
          },
          {
            key: "ui-theme",
            page: "ui-theme",
            label: t("nav.uiTheme"),
            icon: <BgColorsOutlined />,
            roles: all_admin_roles,
          },
        ],
      },
    ],
  },
];

const Sidebar: React.FC<SidebarProps> = ({
  setPage,
  defaultSelectedKey,
  collapsed = false,
  enabledPagesInternalUsers,
  enableProjectsUI,
  disableAgentsForInternalUsers,
  allowAgentsForTeamAdmins,
  disableVectorStoresForInternalUsers,
  allowVectorStoresForTeamAdmins,
}) => {
  const { userId, accessToken, userRole } = useAuthorized();
  const { data: organizations } = useOrganizations();
  const { data: teams } = useTeams();
  const { t } = useTranslation();

  const menuGroups = useMemo(() => getMenuGroups(t), [t]);

  // Check if user is an org_admin
  const isOrgAdmin = useMemo(() => {
    if (!userId || !organizations) return false;
    return organizations.some((org: Organization) =>
      org.members?.some((member) => member.user_id === userId && member.user_role === "org_admin"),
    );
  }, [userId, organizations]);

  // Check if user is a team admin for any team
  const isTeamAdmin = useMemo(() => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""), [teams, userId]);

  // The parent (legacy root page or dashboard layout) owns navigation for both
  // migrated and legacy pages; the sidebar only reports the selected page.
  const navigateToPage = (page: string) => setPage(page);

  // Wrap label in <a> so every nav item supports right-click → "Open in new tab"
  // and Ctrl/Cmd+click to open in a new tab, while preserving SPA navigation for normal clicks.
  const renderNavLink = (label: React.ReactNode, page: string, externalUrl?: string): React.ReactNode => {
    if (externalUrl) {
      return (
        <a
          href={externalUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          style={{ color: "inherit", textDecoration: "none" }}
        >
          {label} <ExportOutlined style={{ fontSize: 10, marginLeft: 4 }} />
        </a>
      );
    }
    const migratedRoute = MIGRATED_PAGES[page];
    const href = migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(page);
    return (
      <a
        href={href}
        onClick={(e) => {
          if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) {
            e.stopPropagation();
            return;
          }
          e.preventDefault();
        }}
        style={{ color: "inherit", textDecoration: "none" }}
      >
        {label}
      </a>
    );
  };

  // Filter items based on user role and enabled pages for internal users
  const filterItemsByRole = (items: MenuItem[]): MenuItem[] => {
    const isAdmin = isAdminRole(userRole);

    // Debug logging
    if (enabledPagesInternalUsers !== null && enabledPagesInternalUsers !== undefined) {
      console.log("[LeftNav] Filtering with enabled pages:", {
        userRole,
        isAdmin,
        enabledPagesInternalUsers,
      });
    }

    return items
      .map((item) => ({
        ...item,
        children: item.children ? filterItemsByRole(item.children) : undefined,
      }))
      .filter((item) => {
        // Special handling for organizations and users menu items - allow org_admins
        if (item.key === "organizations" || item.key === "users") {
          const hasRoleAccess = !item.roles || item.roles.includes(userRole) || isOrgAdmin;
          if (!hasRoleAccess) return false;

          // Check enabled pages for internal users (non-admins)
          if (!isAdmin && enabledPagesInternalUsers !== null && enabledPagesInternalUsers !== undefined) {
            const isIncluded = enabledPagesInternalUsers.includes(item.page);
            console.log(`[LeftNav] Page "${item.page}" (${item.key}): ${isIncluded ? "VISIBLE" : "HIDDEN"}`);
            return isIncluded;
          }
          return true;
        }

        // Hide Projects page if enableProjectsUI is not enabled
        if (item.key === "projects" && !enableProjectsUI) return false;

        // Hide agents and vector-stores pages for non-admin users when disabled,
        // unless allow_*_for_team_admins is on and the user is a team admin.
        if (
          !isAdmin &&
          item.key === "agents" &&
          disableAgentsForInternalUsers &&
          !(allowAgentsForTeamAdmins && isTeamAdmin)
        )
          return false;
        if (
          !isAdmin &&
          item.key === "vector-stores" &&
          disableVectorStoresForInternalUsers &&
          !(allowVectorStoresForTeamAdmins && isTeamAdmin)
        )
          return false;

        // Existing role check
        if (item.roles && !item.roles.includes(userRole)) return false;

        // Check enabled pages for internal users (non-admins)
        if (!isAdmin && enabledPagesInternalUsers !== null && enabledPagesInternalUsers !== undefined) {
          // If item has children, check if any children are visible
          if (item.children && item.children.length > 0) {
            const hasVisibleChildren = item.children.some((child) => enabledPagesInternalUsers.includes(child.page));
            if (hasVisibleChildren) {
              console.log(`[LeftNav] Parent "${item.page}" (${item.key}): VISIBLE (has visible children)`);
              return true;
            }
          }

          const isIncluded = enabledPagesInternalUsers.includes(item.page);
          console.log(`[LeftNav] Page "${item.page}" (${item.key}): ${isIncluded ? "VISIBLE" : "HIDDEN"}`);
          return isIncluded;
        }

        return true;
      });
  };

  // Build menu items with groups
  const buildMenuItems = (): MenuProps["items"] => {
    const items: MenuProps["items"] = [];

    menuGroups.forEach((group) => {
      // Check if group has role restriction
      if (group.roles && !group.roles.includes(userRole)) {
        return;
      }

      const filteredItems = filterItemsByRole(group.items);
      if (filteredItems.length === 0) return;

      // Add group with items
      items.push({
        type: "group",
        label: collapsed ? null : (
          <span
            style={{
              fontSize: "10px",
              fontWeight: 600,
              color: "#6b7280",
              letterSpacing: "0.05em",
              padding: "12px 0 4px 12px",
              display: "block",
              marginBottom: "2px",
            }}
          >
            {group.groupLabel}
          </span>
        ),
        children: filteredItems.map((item) => ({
          key: item.key,
          icon: item.icon,
          label: renderNavLink(item.label, item.page, item.external_url),
          children: item.children?.map((child) => ({
            key: child.key,
            icon: child.icon,
            label: renderNavLink(child.label, child.page, child.external_url),
            onClick: () => {
              if (child.external_url) {
                window.open(child.external_url, "_blank");
              } else {
                navigateToPage(child.page);
              }
            },
          })),
          onClick: !item.children
            ? () => {
                if (item.external_url) {
                  window.open(item.external_url, "_blank");
                } else {
                  navigateToPage(item.page);
                }
              }
            : undefined,
        })),
      });
    });

    return items;
  };

  // Find selected menu key
  const findMenuItemKey = (page: string): string => {
    for (const group of menuGroups) {
      for (const item of group.items) {
        if (item.page === page) return item.key;
        if (item.children) {
          const child = item.children.find((c) => c.page === page);
          if (child) return child.key;
        }
      }
    }
    return "api-keys";
  };

  const selectedMenuKey = findMenuItemKey(defaultSelectedKey);

  return (
    <Layout>
      <Sider
        theme="light"
        width={220}
        collapsed={collapsed}
        collapsedWidth={80}
        collapsible
        trigger={null}
        style={{
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          position: "relative",
        }}
      >
        <ConfigProvider
          theme={{
            components: {
              Menu: {
                iconSize: 15,
                fontSize: 13,
                itemMarginInline: 4,
                itemPaddingInline: 8,
                itemHeight: 30,
                itemBorderRadius: 6,
                subMenuItemBorderRadius: 6,
                groupTitleFontSize: 10,
                groupTitleLineHeight: 1.5,
              },
            },
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedMenuKey]}
            defaultOpenKeys={[]}
            inlineCollapsed={collapsed}
            className="custom-sidebar-menu"
            style={{
              borderRight: 0,
              backgroundColor: "transparent",
              fontSize: "13px",
              paddingTop: "4px",
            }}
            items={buildMenuItems()}
          />
        </ConfigProvider>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </Sider>
    </Layout>
  );
};

export default Sidebar;

export { getMenuGroups };
