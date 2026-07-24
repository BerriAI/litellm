import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useLogout } from "@/app/(dashboard)/hooks/useLogout";
import { getProxyBaseUrl } from "@/components/networking";
import { useTheme } from "@/contexts/ThemeContext";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sidebar,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarSeparator,
  sidebarMenuButtonVariants,
} from "@/components/ui/sidebar";
import {
  Activity,
  BarChart3,
  Bell,
  Blocks,
  Bot,
  BookOpen,
  Building2,
  Boxes,
  ChevronRight,
  Code2,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Folder,
  HeartPulse,
  KeyRound,
  LayoutGrid,
  Network,
  Palette,
  PanelLeftClose,
  PanelLeftOpen,
  PiggyBank,
  PlayCircle,
  Route,
  ScrollText,
  Search,
  Server,
  Settings as SettingsIcon,
  Shield,
  ShieldCheck,
  Tags,
  Terminal,
  User,
  Users,
  Wallet,
  Wrench,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { cn } from "@/lib/cva.config";
import {
  all_admin_roles,
  internalUserRoles,
  isAdminRole,
  isUserTeamAdminForAnyTeam,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
} from "../utils/roles";
import BetaBadge from "./BetaBadge";
import NewBadge from "./common_components/NewBadge";
import type { Organization } from "./networking";
import SidebarAccountMenu from "./SidebarAccountMenu/SidebarAccountMenu";
import SidebarUsageCard from "./SidebarUsageCard";
import { MIGRATED_PAGES, migratedHref, legacyPageHref } from "@/utils/migratedPages";

const ICON = { strokeWidth: 1.75 } as const;

interface SidebarProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
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

// Menu groups organized by category - defined outside component for export.
// Shape (key/page/label/roles/children) is consumed by page_utils.ts; only the
// icons changed to lucide as part of the sidebar redesign.
const menuGroups: MenuGroup[] = [
  {
    groupLabel: "AI GATEWAY",
    items: [
      { key: "api-keys", page: "api-keys", label: "Virtual Keys", icon: <KeyRound {...ICON} /> },
      {
        key: "llm-playground",
        page: "llm-playground",
        label: "Playground",
        icon: <PlayCircle {...ICON} />,
        roles: rolesWithWriteAccess,
      },
      {
        key: "models",
        page: "models",
        label: "Models + Endpoints",
        icon: <Network {...ICON} />,
        roles: rolesAllowedToViewWriteScopedPages,
      },
      {
        key: "agentic",
        page: "agentic",
        label: "Agentic",
        icon: <Bot {...ICON} />,
        children: [
          {
            key: "agents",
            page: "agents",
            label: "Agents",
            icon: <Bot {...ICON} />,
            roles: rolesAllowedToViewWriteScopedPages,
          },
          { key: "workflows", page: "workflows", label: "Workflow Runs", icon: <Workflow {...ICON} /> },
          { key: "memory", page: "memory", label: "Memory", icon: <Database {...ICON} /> },
        ],
      },
      { key: "mcp-servers", page: "mcp-servers", label: "MCP Servers", icon: <Server {...ICON} /> },
      { key: "skills", page: "skills", label: "Skills", icon: <Blocks {...ICON} />, roles: all_admin_roles },
      { key: "guardrails", page: "guardrails", label: "Guardrails", icon: <Shield {...ICON} /> },
      {
        key: "policies",
        page: "policies",
        label: "Policies",
        icon: <ScrollText {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "tools",
        page: "tools",
        label: "Tools",
        icon: <Wrench {...ICON} />,
        children: [
          { key: "search-tools", page: "search-tools", label: "Search Tools", icon: <Search {...ICON} /> },
          { key: "vector-stores", page: "vector-stores", label: "Vector Stores", icon: <Database {...ICON} /> },
          { key: "tool-policies", page: "tool-policies", label: "Tool Policies", icon: <ShieldCheck {...ICON} /> },
        ],
      },
    ],
  },
  {
    groupLabel: "OBSERVABILITY",
    items: [
      {
        key: "new_usage",
        page: "new_usage",
        icon: <BarChart3 {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: "Usage",
      },
      {
        key: "cost-optimization",
        page: "cost-optimization",
        icon: <PiggyBank {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: "Cost Optimization",
      },
      { key: "logs", page: "logs", label: "Logs", icon: <Activity {...ICON} /> },
      {
        key: "guardrails-monitor",
        page: "guardrails-monitor",
        label: "Guardrails Monitor",
        icon: <HeartPulse {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles],
      },
    ],
  },
  {
    groupLabel: "ACCESS CONTROL",
    items: [
      { key: "teams", page: "teams", label: "Teams", icon: <Users {...ICON} /> },
      {
        key: "projects",
        page: "projects",
        label: (
          <span className="flex items-center gap-2">
            Projects <BetaBadge />
          </span>
        ),
        icon: <Folder {...ICON} />,
        roles: all_admin_roles,
      },
      { key: "users", page: "users", label: "Internal Users", icon: <User {...ICON} />, roles: all_admin_roles },
      {
        key: "organizations",
        page: "organizations",
        label: "Organizations",
        icon: <Building2 {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "access-groups",
        page: "access-groups",
        label: "Access Groups",
        icon: <Boxes {...ICON} />,
        roles: all_admin_roles,
      },
      { key: "budgets", page: "budgets", label: "Budgets", icon: <Wallet {...ICON} />, roles: all_admin_roles },
    ],
  },
  {
    groupLabel: "DEVELOPER TOOLS",
    items: [
      { key: "api_ref", page: "api_ref", label: "API Reference", icon: <Code2 {...ICON} /> },
      { key: "model-hub-table", page: "model-hub-table", label: "AI Hub", icon: <LayoutGrid {...ICON} /> },
      {
        key: "learning-resources",
        page: "learning-resources",
        label: "Learning Resources",
        icon: <BookOpen {...ICON} />,
        external_url: "https://models.litellm.ai/cookbook",
      },
      {
        key: "caching",
        page: "caching",
        label: "Response Cache",
        icon: <Database {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "experimental",
        page: "experimental",
        label: "Experimental",
        icon: <FlaskConical {...ICON} />,
        children: [
          { key: "prompts", page: "prompts", label: "Prompts", icon: <FileText {...ICON} />, roles: all_admin_roles },
          {
            key: "transform-request",
            page: "transform-request",
            label: "API Playground",
            icon: <Terminal {...ICON} />,
            roles: [...all_admin_roles, ...internalUserRoles],
          },
          {
            key: "tag-management",
            page: "tag-management",
            label: "Tag Management",
            icon: <Tags {...ICON} />,
            roles: all_admin_roles,
          },
          { key: "4", page: "usage", label: "Old Usage", icon: <BarChart3 {...ICON} /> },
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
        label: (
          <span className="flex items-center gap-2">
            Settings <NewBadge />
          </span>
        ),
        icon: <SettingsIcon {...ICON} />,
        roles: all_admin_roles,
        children: [
          {
            key: "router-settings",
            page: "router-settings",
            label: "Router Settings",
            icon: <Route {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "logging-and-alerts",
            page: "logging-and-alerts",
            label: "Logging & Alerts",
            icon: <Bell {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "admin-panel",
            page: "admin-panel",
            label: (
              <span className="flex items-center gap-2">
                Admin Settings{" "}
                <NewBadge dot>
                  <span />
                </NewBadge>
              </span>
            ),
            icon: <SettingsIcon {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "cost-tracking",
            page: "cost-tracking",
            label: "Cost Tracking",
            icon: <BarChart3 {...ICON} />,
            roles: all_admin_roles,
          },
          { key: "ui-theme", page: "ui-theme", label: "UI Theme", icon: <Palette {...ICON} />, roles: all_admin_roles },
        ],
      },
    ],
  },
];

const findParentKey = (page: string): string | null => {
  for (const group of menuGroups) {
    for (const item of group.items) {
      if (item.children?.some((c) => c.page === page || c.key === page)) return item.key;
    }
  }
  return null;
};

const findMenuItemKey = (page: string): string => {
  for (const group of menuGroups) {
    for (const item of group.items) {
      if (item.page === page) return item.key;
      const child = item.children?.find((c) => c.page === page);
      if (child) return child.key;
    }
  }
  return "api-keys";
};

const labelText = (item: MenuItem): string => (typeof item.label === "string" ? item.label : item.key);

const SECTION_DISPLAY: Record<string, string> = {
  "AI GATEWAY": "AI Gateway",
  OBSERVABILITY: "Observability",
  "ACCESS CONTROL": "Access Control",
  "DEVELOPER TOOLS": "Developer Tools",
  SETTINGS: "Settings",
};

const prettify = (key: string): string =>
  key
    .split(/[-_]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

// Breadcrumb ("Section" / "Page") for the top bar, derived from the same nav config.
export const getBreadcrumb = (page: string): { section: string | null; title: string } => {
  for (const group of menuGroups) {
    for (const item of group.items) {
      const section = SECTION_DISPLAY[group.groupLabel] ?? group.groupLabel;
      if (item.page === page)
        return { section, title: typeof item.label === "string" ? item.label : prettify(item.key) };
      const child = item.children?.find((c) => c.page === page);
      if (child) return { section, title: typeof child.label === "string" ? child.label : prettify(child.key) };
    }
  }
  return { section: null, title: prettify(page) };
};

const Sidebar_: React.FC<SidebarProps> = ({
  setPage,
  defaultSelectedKey,
  collapsed = false,
  onToggleCollapsed,
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
  const { logoUrl } = useTheme();
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const logout = useLogout(accessToken);

  const baseUrl = getProxyBaseUrl();
  const version = healthData?.litellm_version;
  const selectedKey = findMenuItemKey(defaultSelectedKey);

  const [openGroups, setOpenGroups] = useState<Set<string>>(() => {
    const parent = findParentKey(defaultSelectedKey);
    return new Set(parent ? [parent] : []);
  });

  // Keep the active page's parent group expanded as the user navigates, using the
  // "adjust state during render" pattern rather than an effect (avoids a
  // setState-in-effect render cascade).
  const [prevSelectedKey, setPrevSelectedKey] = useState(defaultSelectedKey);
  if (defaultSelectedKey !== prevSelectedKey) {
    setPrevSelectedKey(defaultSelectedKey);
    const parent = findParentKey(defaultSelectedKey);
    if (parent && !openGroups.has(parent)) {
      setOpenGroups((prev) => new Set(prev).add(parent));
    }
  }

  const isOrgAdmin = useMemo(() => {
    if (!userId || !organizations) return false;
    return organizations.some((org: Organization) =>
      org.members?.some((member) => member.user_id === userId && member.user_role === "org_admin"),
    );
  }, [userId, organizations]);

  const isTeamAdmin = useMemo(() => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""), [teams, userId]);

  const filterItemsByRole = (items: MenuItem[]): MenuItem[] => {
    const isAdmin = isAdminRole(userRole);
    return items
      .map((item) => ({ ...item, children: item.children ? filterItemsByRole(item.children) : undefined }))
      .filter((item) => {
        if (item.key === "organizations" || item.key === "users") {
          const hasRoleAccess = !item.roles || item.roles.includes(userRole) || isOrgAdmin;
          if (!hasRoleAccess) return false;
          if (!isAdmin && enabledPagesInternalUsers != null) return enabledPagesInternalUsers.includes(item.page);
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
        if (!isAdmin && enabledPagesInternalUsers != null) {
          if (item.children && item.children.length > 0) {
            const hasVisibleChildren = item.children.some((child) => enabledPagesInternalUsers.includes(child.page));
            if (hasVisibleChildren) return true;
          }
          return enabledPagesInternalUsers.includes(item.page);
        }
        return true;
      });
  };

  const visibleGroups = menuGroups
    .filter((group) => !group.roles || group.roles.includes(userRole))
    .map((group) => ({ groupLabel: group.groupLabel, items: filterItemsByRole(group.items) }))
    .filter((group) => group.items.length > 0);

  const toggleGroup = (key: string) => {
    if (collapsed) {
      onToggleCollapsed?.();
      setOpenGroups((prev) => new Set(prev).add(key));
      return;
    }
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const handleLeafClick = (e: React.MouseEvent, item: MenuItem) => {
    if (item.external_url) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
    e.preventDefault();
    setPage(item.page);
  };

  const renderLeaf = (item: MenuItem, isChild: boolean) => {
    const active = selectedKey === item.key;
    const size = isChild ? "sub" : "default";
    const label = <span className="flex-1 truncate group-data-[collapsed=true]/sidebar:hidden">{item.label}</span>;

    if (item.external_url) {
      return (
        <a
          key={item.key}
          href={item.external_url}
          target="_blank"
          rel="noopener noreferrer"
          title={collapsed ? labelText(item) : undefined}
          data-active={active || undefined}
          className={cn(sidebarMenuButtonVariants({ isActive: active, size }))}
        >
          {item.icon}
          {label}
          <ExternalLink className="size-3.5 shrink-0 opacity-70 group-data-[collapsed=true]/sidebar:hidden" />
        </a>
      );
    }

    const href = MIGRATED_PAGES[item.page] ? migratedHref(MIGRATED_PAGES[item.page]) : legacyPageHref(item.page);
    return (
      <a
        key={item.key}
        href={href}
        onClick={(e) => handleLeafClick(e, item)}
        title={collapsed ? labelText(item) : undefined}
        data-active={active || undefined}
        className={cn(sidebarMenuButtonVariants({ isActive: active, size }))}
      >
        {item.icon}
        {label}
      </a>
    );
  };

  const renderItem = (item: MenuItem) => {
    const isGroup = !!item.children && item.children.length > 0;
    if (!isGroup) {
      return <SidebarMenuItem key={item.key}>{renderLeaf(item, false)}</SidebarMenuItem>;
    }

    const active = selectedKey === item.key;
    const open = openGroups.has(item.key);
    return (
      <SidebarMenuItem key={item.key}>
        <SidebarMenuButton
          isActive={active}
          onClick={() => toggleGroup(item.key)}
          title={collapsed ? labelText(item) : undefined}
        >
          {item.icon}
          <span className="flex-1 truncate group-data-[collapsed=true]/sidebar:hidden">{item.label}</span>
          <ChevronRight
            className={cn(
              "size-4 shrink-0 transition-transform group-data-[collapsed=true]/sidebar:hidden",
              open && "rotate-90",
            )}
          />
        </SidebarMenuButton>
        {open && (
          <SidebarMenuSub>
            {item.children!.map((child) => (
              <SidebarMenuItem key={child.key}>{renderLeaf(child, true)}</SidebarMenuItem>
            ))}
          </SidebarMenuSub>
        )}
      </SidebarMenuItem>
    );
  };

  const logoSrc = logoUrl || `${baseUrl}/get_image`;

  return (
    <Sidebar collapsed={collapsed}>
      <SidebarHeader className="h-14 border-b border-border group-data-[collapsed=true]/sidebar:h-auto">
        <div className="flex items-center justify-between gap-2 group-data-[collapsed=true]/sidebar:flex-col">
          <div className="flex min-w-0 items-center gap-2">
            <Link href={baseUrl || "/"} className="flex min-w-0 items-center" aria-label="LiteLLM home">
              <img
                src={logoSrc}
                alt="LiteLLM"
                className="h-7 w-auto max-w-[150px] object-contain group-data-[collapsed=true]/sidebar:w-7"
              />
            </Link>
            {version && (
              <Badge
                variant="outline"
                render={<a href="https://docs.litellm.ai/release_notes" target="_blank" rel="noopener noreferrer" />}
                className="px-1.5 py-0 font-mono text-[10px] font-medium text-muted-foreground group-data-[collapsed=true]/sidebar:hidden"
              >
                v{version}
              </Badge>
            )}
          </div>
          {onToggleCollapsed && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onToggleCollapsed}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              className="flex-none text-muted-foreground"
            >
              {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            </Button>
          )}
        </div>
      </SidebarHeader>

      <ScrollArea className="min-h-0 flex-1">
        <nav className="flex flex-col gap-0.5 px-3 pb-3">
          {visibleGroups.map((group, gi) => (
            <SidebarGroup key={group.groupLabel}>
              {gi > 0 && <SidebarSeparator className="hidden group-data-[collapsed=true]/sidebar:block" />}
              <SidebarGroupLabel>{group.groupLabel}</SidebarGroupLabel>
              <SidebarMenu>{group.items.map((item) => renderItem(item))}</SidebarMenu>
            </SidebarGroup>
          ))}
        </nav>
      </ScrollArea>

      <SidebarFooter>
        {isAdminRole(userRole) && (
          <SidebarUsageCard
            accessToken={accessToken}
            collapsed={collapsed}
            onExpandRail={() => onToggleCollapsed?.()}
          />
        )}
        <SidebarAccountMenu onLogout={logout} collapsed={collapsed} />
      </SidebarFooter>
    </Sidebar>
  );
};

export default Sidebar_;

export { menuGroups };
