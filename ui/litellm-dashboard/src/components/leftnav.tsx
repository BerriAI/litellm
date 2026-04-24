import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  Plug as ApiOutlined,
  LayoutGrid as AppstoreOutlined,
  ClipboardCheck as AuditOutlined,
  Landmark as BankOutlined,
  BarChart3 as BarChartOutlined,
  Palette as BgColorsOutlined,
  Square as BlockOutlined,
  Book as BookOutlined,
  ChevronDown,
  CreditCard as CreditCardOutlined,
  Database as DatabaseOutlined,
  FlaskConical as ExperimentOutlined,
  ExternalLink as ExportOutlined,
  FileText as FileTextOutlined,
  Folder as FolderOutlined,
  Key as KeyOutlined,
  LineChart as LineChartOutlined,
  PlayCircle as PlayCircleOutlined,
  Bot as RobotOutlined,
  ShieldCheck as SafetyOutlined,
  Search as SearchOutlined,
  Settings as SettingOutlined,
  Tags as TagsOutlined,
  Users as TeamOutlined,
  Wrench as ToolOutlined,
  User as UserOutlined,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  all_admin_roles,
  internalUserRoles,
  isAdminRole,
  isUserTeamAdminForAnyTeam,
  rolesWithWriteAccess,
} from "../utils/roles";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import NewBadge from "./common_components/NewBadge";
import type { Organization } from "./networking";
import UsageIndicator from "./UsageIndicator";
import { serverRootPath } from "./networking";

/**
 * Pages migrated to path-based routing under (dashboard)/.
 * Key = legacy page id, Value = route segment.
 */
const MIGRATED_PAGES: Record<string, string> = {
  "api-reference": "api-reference",
};

/** Build an absolute href for a migrated page. */
function migratedHref(routeSegment: string): string {
  const raw = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = raw.replace(/^\/+|\/+$/g, "");
  let base = trimmed ? `/${trimmed}/` : "/";

  if (serverRootPath && serverRootPath !== "/") {
    const cleanRoot = serverRootPath.replace(/\/+$/, "");
    const cleanBase = base.replace(/^\/+/, "");
    base = `${cleanRoot}/${cleanBase}`;
  }

  return `${base}${routeSegment}`;
}

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

interface MenuItem {
  key: string;
  page: string;
  label: string | React.ReactNode;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
  external_url?: string;
}

interface MenuGroup {
  groupLabel: string;
  items: MenuItem[];
  roles?: string[];
}

const menuGroups: MenuGroup[] = [
  {
    groupLabel: "AI GATEWAY",
    items: [
      { key: "api-keys", page: "api-keys", label: "Virtual Keys", icon: <KeyOutlined /> },
      { key: "llm-playground", page: "llm-playground", label: "Playground", icon: <PlayCircleOutlined />, roles: rolesWithWriteAccess },
      { key: "models", page: "models", label: "Models + Endpoints", icon: <BlockOutlined />, roles: rolesWithWriteAccess },
      { key: "agents", page: "agents", label: "Agents", icon: <RobotOutlined />, roles: rolesWithWriteAccess },
      { key: "mcp-servers", page: "mcp-servers", label: "MCP Servers", icon: <ToolOutlined /> },
      { key: "skills", page: "skills", label: "Skills", icon: <ApiOutlined />, roles: all_admin_roles },
      { key: "guardrails", page: "guardrails", label: "Guardrails", icon: <SafetyOutlined /> },
      {
        key: "policies",
        page: "policies",
        label: <span className="flex items-center gap-4">Policies</span>,
        icon: <AuditOutlined />,
        roles: all_admin_roles,
      },
      {
        key: "tools",
        page: "tools",
        label: "Tools",
        icon: <ToolOutlined />,
        children: [
          { key: "search-tools", page: "search-tools", label: "Search Tools", icon: <SearchOutlined /> },
          { key: "vector-stores", page: "vector-stores", label: "Vector Stores", icon: <DatabaseOutlined /> },
          { key: "tool-policies", page: "tool-policies", label: "Tool Policies", icon: <SafetyOutlined /> },
        ],
      },
    ],
  },
  {
    groupLabel: "OBSERVABILITY",
    items: [
      { key: "new_usage", page: "new_usage", icon: <BarChartOutlined />, roles: [...all_admin_roles, ...internalUserRoles], label: "Usage" },
      { key: "logs", page: "logs", label: "Logs", icon: <LineChartOutlined /> },
      { key: "guardrails-monitor", page: "guardrails-monitor", label: "Guardrails Monitor", icon: <SafetyOutlined />, roles: [...all_admin_roles, ...internalUserRoles] },
    ],
  },
  {
    groupLabel: "ACCESS CONTROL",
    items: [
      { key: "teams", page: "teams", label: "Teams", icon: <TeamOutlined /> },
      {
        key: "projects",
        page: "projects",
        label: <span className="flex items-center gap-2">Projects <NewBadge /></span>,
        icon: <FolderOutlined />,
        roles: all_admin_roles,
      },
      { key: "users", page: "users", label: "Internal Users", icon: <UserOutlined />, roles: all_admin_roles },
      { key: "organizations", page: "organizations", label: "Organizations", icon: <BankOutlined />, roles: all_admin_roles },
      { key: "access-groups", page: "access-groups", label: "Access Groups", icon: <BlockOutlined />, roles: all_admin_roles },
      { key: "budgets", page: "budgets", label: "Budgets", icon: <CreditCardOutlined />, roles: all_admin_roles },
    ],
  },
  {
    groupLabel: "DEVELOPER TOOLS",
    items: [
      { key: "api-reference", page: "api-reference", label: "API Reference", icon: <ApiOutlined /> },
      { key: "model-hub-table", page: "model-hub-table", label: "AI Hub", icon: <AppstoreOutlined /> },
      { key: "learning-resources", page: "learning-resources", label: "Learning Resources", icon: <BookOutlined />, external_url: "https://models.litellm.ai/cookbook" },
      {
        key: "experimental",
        page: "experimental",
        label: "Experimental",
        icon: <ExperimentOutlined />,
        children: [
          { key: "caching", page: "caching", label: "Caching", icon: <DatabaseOutlined />, roles: all_admin_roles },
          { key: "prompts", page: "prompts", label: "Prompts", icon: <FileTextOutlined />, roles: all_admin_roles },
          { key: "transform-request", page: "transform-request", label: "API Playground", icon: <ApiOutlined />, roles: [...all_admin_roles, ...internalUserRoles] },
          { key: "tag-management", page: "tag-management", label: "Tag Management", icon: <TagsOutlined />, roles: all_admin_roles },
          { key: "4", page: "usage", label: "Old Usage", icon: <BarChartOutlined /> },
        ],
      },
    ],
  },
  {
    groupLabel: "SETTINGS",
    roles: all_admin_roles,
    items: [
      {
        key: "settings",
        page: "settings",
        label: <span className="flex items-center gap-2">Settings <NewBadge /></span>,
        icon: <SettingOutlined />,
        roles: all_admin_roles,
        children: [
          { key: "router-settings", page: "router-settings", label: "Router Settings", icon: <SettingOutlined />, roles: all_admin_roles },
          { key: "logging-and-alerts", page: "logging-and-alerts", label: "Logging & Alerts", icon: <SettingOutlined />, roles: all_admin_roles },
          {
            key: "admin-panel",
            page: "admin-panel",
            label: (
              <span className="flex items-center gap-2">
                Admin Settings <NewBadge dot><span /></NewBadge>
              </span>
            ),
            icon: <SettingOutlined />,
            roles: all_admin_roles,
          },
          { key: "cost-tracking", page: "cost-tracking", label: "Cost Tracking", icon: <BarChartOutlined />, roles: all_admin_roles },
          { key: "ui-theme", page: "ui-theme", label: "UI Theme", icon: <BgColorsOutlined />, roles: all_admin_roles },
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

  const isOrgAdmin = useMemo(() => {
    if (!userId || !organizations) return false;
    return organizations.some((org: Organization) =>
      org.members?.some(
        (member) => member.user_id === userId && member.user_role === "org_admin",
      ),
    );
  }, [userId, organizations]);

  const isTeamAdmin = useMemo(
    () => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""),
    [teams, userId],
  );

  // Auto-open any parent submenu that contains the currently selected page.
  const initialOpenKeys = useMemo(() => {
    const open: string[] = [];
    for (const group of menuGroups) {
      for (const item of group.items) {
        if (item.children?.some((c) => c.page === defaultSelectedKey)) {
          open.push(item.key);
        }
      }
    }
    return open;
  }, [defaultSelectedKey]);

  const [openKeys, setOpenKeys] = useState<string[]>(initialOpenKeys);

  const toggleKey = (key: string) => {
    setOpenKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const navigateToPage = (page: string) => {
    if (MIGRATED_PAGES[page]) {
      setPage(page);
      return;
    }
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("page", page);
    window.history.pushState(null, "", `?${newSearchParams.toString()}`);
    setPage(page);
  };

  const filterItemsByRole = (items: MenuItem[]): MenuItem[] => {
    const isAdmin = isAdminRole(userRole);

    return items
      .map((item) => ({
        ...item,
        children: item.children ? filterItemsByRole(item.children) : undefined,
      }))
      .filter((item) => {
        if (item.key === "organizations" || item.key === "users") {
          const hasRoleAccess =
            !item.roles || item.roles.includes(userRole) || isOrgAdmin;
          if (!hasRoleAccess) return false;
          if (
            !isAdmin &&
            enabledPagesInternalUsers !== null &&
            enabledPagesInternalUsers !== undefined
          ) {
            return enabledPagesInternalUsers.includes(item.page);
          }
          return true;
        }
        if (item.key === "projects" && !enableProjectsUI) return false;
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
        if (item.roles && !item.roles.includes(userRole)) return false;

        if (
          !isAdmin &&
          enabledPagesInternalUsers !== null &&
          enabledPagesInternalUsers !== undefined
        ) {
          if (item.children && item.children.length > 0) {
            const hasVisibleChildren = item.children.some((child) =>
              enabledPagesInternalUsers.includes(child.page),
            );
            if (hasVisibleChildren) return true;
          }
          return enabledPagesInternalUsers.includes(item.page);
        }

        return true;
      });
  };

  // Build href: either migrated-route or ?page=... query.
  const hrefFor = (item: MenuItem): string => {
    if (item.external_url) return item.external_url;
    const migratedRoute = MIGRATED_PAGES[item.page];
    if (migratedRoute) return migratedHref(migratedRoute);
    if (typeof window === "undefined") return `?page=${item.page}`;
    const params = new URLSearchParams(window.location.search);
    params.set("page", item.page);
    return `?${params.toString()}`;
  };

  const renderItem = (item: MenuItem, depth: number) => {
    const isSelected = item.page === defaultSelectedKey;
    const hasChildren = item.children && item.children.length > 0;
    const isOpen = openKeys.includes(item.key);
    const indent = depth * 12;

    const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
      e.preventDefault();
      if (item.external_url) {
        window.open(item.external_url, "_blank");
        return;
      }
      if (hasChildren) {
        toggleKey(item.key);
      } else {
        navigateToPage(item.page);
      }
    };

    const linkContent = (
      <>
        {item.icon && (
          <span
            className={cn(
              "inline-flex items-center justify-center shrink-0",
              "h-[15px] w-[15px]",
              "[&>svg]:h-[15px] [&>svg]:w-[15px]",
            )}
          >
            {item.icon}
          </span>
        )}
        {!collapsed && (
          <span className="flex-1 truncate">
            {item.label}
            {item.external_url && (
              <ExportOutlined className="h-2.5 w-2.5 ml-1 inline" />
            )}
          </span>
        )}
        {!collapsed && hasChildren && (
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 transition-transform text-muted-foreground",
              isOpen && "rotate-180",
            )}
          />
        )}
      </>
    );

    const itemClasses = cn(
      "flex items-center gap-2 rounded-md text-[13px] px-2 mx-1 h-[30px]",
      "transition-colors cursor-pointer",
      "text-foreground no-underline",
      isSelected
        ? "bg-primary/10 text-primary font-medium"
        : "hover:bg-muted",
    );

    const link = (
      <a
        href={hrefFor(item)}
        onClick={handleClick}
        style={{ paddingLeft: `${8 + indent}px` }}
        className={itemClasses}
      >
        {linkContent}
      </a>
    );

    // When collapsed, wrap icon with a tooltip that shows the label.
    const wrapped = collapsed ? (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{link}</TooltipTrigger>
          <TooltipContent side="right">{item.label}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    ) : (
      link
    );

    return (
      <li key={item.key} role="none">
        {wrapped}
        {hasChildren && !collapsed && isOpen && (
          <ul role="group" className="mt-[2px] mb-[2px]">
            {item.children!.map((child) => renderItem(child, depth + 1))}
          </ul>
        )}
      </li>
    );
  };

  const filteredGroups = menuGroups
    .filter((group) => !group.roles || group.roles.includes(userRole))
    .map((group) => ({
      ...group,
      items: filterItemsByRole(group.items),
    }))
    .filter((group) => group.items.length > 0);

  return (
    <aside
      className={cn(
        "relative bg-background",
        "transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]",
      )}
      style={{ width: collapsed ? 80 : 220 }}
    >
      <nav
        aria-label="Main navigation"
        className="custom-sidebar-menu pt-1"
      >
        <ul role="menu" className="list-none p-0 m-0">
          {filteredGroups.map((group) => (
            <li key={group.groupLabel} role="none">
              {!collapsed && (
                <div className="text-[10px] font-semibold text-muted-foreground tracking-wider pt-3 pb-1 pl-3 mb-[2px]">
                  {group.groupLabel}
                </div>
              )}
              <ul role="group" className="list-none p-0 m-0">
                {group.items.map((item) => renderItem(item, 0))}
              </ul>
            </li>
          ))}
        </ul>
      </nav>
      {isAdminRole(userRole) && !collapsed && (
        <UsageIndicator accessToken={accessToken} width={220} />
      )}
    </aside>
  );
};

export default Sidebar;

// Also export menuGroups for advanced use cases
export { menuGroups };
