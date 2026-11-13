import { SidebarBrand } from "@/components/shared/SidebarBrand";
import { canViewProjectsPage, projectReaderRoles } from "@/app/(dashboard)/hooks/projects/projectAccess";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { useLogout } from "@/app/(dashboard)/hooks/useLogout";
import { useTerminology } from "@/contexts/TerminologyContext";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sidebar,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarSeparator,
  sidebarMenuButtonVariants,
} from "@/components/shared/Sidebar";
import {
  Activity,
  Aperture,
  BarChart3,
  Bell,
  Blocks,
  Bot,
  Building2,
  Boxes,
  ChevronRight,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Folder,
  HeartPulse,
  House,
  KeyRound,
  LayoutGrid,
  Network,
  Palette,
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
import ViewSwitcher from "@/components/Navbar/ViewSwitcher";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { AriaSidebarLink } from "@/components/liteadmin/LiteAdmin";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import type { ParseKeys } from "i18next";
import i18n from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";
import { rolesWithCapability } from "../utils/capabilities";
import {
  all_admin_roles,
  formatUserRole,
  internalUserRoles,
  isAdminRole,
  isUserTeamAdminForAnyTeam,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
  proxyAdminTierRoles,
} from "../utils/roles";
import BetaBadge from "./BetaBadge";
import SidebarAccountMenu from "./SidebarAccountMenu/SidebarAccountMenu";
import SidebarUsageCard from "./SidebarUsageCard";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";

const ICON = { strokeWidth: 1.75 } as const;

interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
  enabledPagesInternalUsers?: string[] | null;
  enableProjectsUI?: boolean;
  disableAgentsForInternalUsers?: boolean;
  allowAgentsForTeamAdmins?: boolean;
  disableVectorStoresForInternalUsers?: boolean;
  allowVectorStoresForTeamAdmins?: boolean;
}

export interface MenuItem {
  key: string;
  page: string;
  route?: string;
  label: string | React.ReactNode;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
  external_url?: string;
}

export interface MenuGroup {
  groupLabel: string;
  items: MenuItem[];
  roles?: string[];
}

export interface MenuVisibilityContext {
  userRole: string;
  isViewOnly: boolean;
  isOrgAdmin: boolean;
  isTeamAdmin: boolean;
  enabledPagesInternalUsers?: string[] | null;
  enableProjectsUI?: boolean;
  disableAgentsForInternalUsers?: boolean;
  allowAgentsForTeamAdmins?: boolean;
  disableVectorStoresForInternalUsers?: boolean;
  allowVectorStoresForTeamAdmins?: boolean;
}

const adminPageVisibility = (item: MenuItem, context: MenuVisibilityContext, isAdmin: boolean): boolean | null => {
  if (item.key !== "organizations" && item.key !== "users") return null;
  const hasRoleAccess = !item.roles || item.roles.includes(context.userRole) || context.isOrgAdmin;
  if (!hasRoleAccess) return false;
  if (!isAdmin && context.enabledPagesInternalUsers != null) {
    return context.enabledPagesInternalUsers.includes(item.page);
  }
  return true;
};

const projectPageIsVisible = (item: MenuItem, context: MenuVisibilityContext): boolean => {
  if (item.key !== "projects") return true;
  if (!context.enableProjectsUI) return false;
  return canViewProjectsPage({
    userRole: context.userRole,
    isOrgAdmin: context.isOrgAdmin,
    isTeamAdmin: context.isTeamAdmin,
  });
};

const pageIsDisabledForInternalUsers = (
  item: MenuItem,
  pageKey: "agents" | "vector-stores",
  isAdmin: boolean,
  context: MenuVisibilityContext,
): boolean => {
  const isDisabled =
    pageKey === "agents" ? context.disableAgentsForInternalUsers : context.disableVectorStoresForInternalUsers;
  const allowedForTeamAdmins =
    pageKey === "agents" ? context.allowAgentsForTeamAdmins : context.allowVectorStoresForTeamAdmins;
  if (item.key !== pageKey || isAdmin || !isDisabled) return false;
  return !(allowedForTeamAdmins && context.isTeamAdmin);
};

const pageIsEnabledForInternalUser = (item: MenuItem, context: MenuVisibilityContext): boolean => {
  const enabledPages = context.enabledPagesInternalUsers;
  if (enabledPages == null) return true;
  if (item.children?.some((child) => enabledPages.includes(child.page))) return true;
  return enabledPages.includes(item.page);
};

const menuItemIsVisible = (item: MenuItem, context: MenuVisibilityContext, isAdmin: boolean): boolean => {
  if (item.children && item.children.length === 0) return false;
  if (item.key === "llm-playground" && context.isViewOnly) return false;
  const adminVisibility = adminPageVisibility(item, context, isAdmin);
  if (adminVisibility !== null) return adminVisibility;
  if (!projectPageIsVisible(item, context)) return false;
  if (pageIsDisabledForInternalUsers(item, "agents", isAdmin, context)) return false;
  if (pageIsDisabledForInternalUsers(item, "vector-stores", isAdmin, context)) return false;
  if (item.roles && !item.roles.includes(context.userRole)) return false;
  if (!isAdmin && context.enabledPagesInternalUsers != null) return pageIsEnabledForInternalUser(item, context);
  return true;
};

export const visibleMenuGroups = (groups: readonly MenuGroup[], context: MenuVisibilityContext): MenuGroup[] => {
  const isAdmin = isAdminRole(context.userRole);
  const filterItems = (items: MenuItem[]): MenuItem[] =>
    items
      .map((item) => ({ ...item, children: item.children ? filterItems(item.children) : undefined }))
      .filter((item) => menuItemIsVisible(item, context, isAdmin));

  return groups
    .filter((group) => !group.roles || group.roles.includes(context.userRole))
    .map((group) => ({ groupLabel: group.groupLabel, items: filterItems(group.items) }))
    .filter((group) => group.items.length > 0);
};

// Menu groups organized by category - defined outside component for export.
// Shape (key/page/label/roles/children) is consumed by page_utils.ts; only the
// icons changed to lucide as part of the sidebar redesign.
const customerPortalGroups: MenuGroup[] = [
  {
    groupLabel: "CUSTOMER PORTAL",
    items: [
      {
        key: "customers",
        page: "customers",
        label: "shell.nav.items.customers",
        icon: <Users {...ICON} />,
        roles: all_admin_roles,
      },
    ],
  },
];

const menuGroups: MenuGroup[] = [
  {
    groupLabel: "AI GATEWAY",
    items: [
      { key: "home", page: "home", label: "shell.nav.items.home", icon: <House {...ICON} /> },
      { key: "api-keys", page: "api-keys", label: "shell.nav.items.virtualKeys", icon: <KeyRound {...ICON} /> },
      {
        key: "llm-playground",
        page: "llm-playground",
        route: "playground",
        label: "shell.nav.items.playground",
        icon: <PlayCircle {...ICON} />,
        roles: rolesWithWriteAccess,
      },
      {
        key: "models",
        page: "models",
        route: "models-and-endpoints",
        label: "shell.nav.items.modelsAndEndpoints",
        icon: <Network {...ICON} />,
        roles: rolesAllowedToViewWriteScopedPages,
      },
      {
        key: "agentic",
        page: "agentic",
        label: "shell.nav.items.agentic",
        icon: <Bot {...ICON} />,
        children: [
          {
            key: "agents",
            page: "agents",
            label: "shell.nav.items.agents",
            icon: <Bot {...ICON} />,
            roles: rolesAllowedToViewWriteScopedPages,
          },
          {
            key: "workflows",
            page: "workflows",
            label: "shell.nav.items.workflowRuns",
            icon: <Workflow {...ICON} />,
            roles: rolesWithCapability("viewWorkflowRuns"),
          },
          {
            key: "memory",
            page: "memory",
            label: "shell.nav.items.memory",
            icon: <Database {...ICON} />,
            roles: rolesWithCapability("viewMemory"),
          },
        ],
      },
      { key: "mcp-servers", page: "mcp-servers", label: "shell.nav.items.mcpServers", icon: <Server {...ICON} /> },
      {
        key: "skills",
        page: "skills",
        label: "shell.nav.items.skills",
        icon: <Blocks {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles, formatUserRole("org_admin")],
      },
      { key: "guardrails", page: "guardrails", label: "shell.nav.items.guardrails", icon: <Shield {...ICON} /> },
      {
        key: "policies",
        page: "policies",
        label: "shell.nav.items.policies",
        icon: <ScrollText {...ICON} />,
        roles: rolesWithCapability("viewPolicies"),
      },
      {
        key: "tools",
        page: "tools",
        label: "shell.nav.items.tools",
        icon: <Wrench {...ICON} />,
        children: [
          {
            key: "search-tools",
            page: "search-tools",
            label: "shell.nav.items.searchTools",
            icon: <Search {...ICON} />,
          },
          {
            key: "vector-stores",
            page: "vector-stores",
            label: "shell.nav.items.vectorStores",
            icon: <Database {...ICON} />,
          },
          {
            key: "tool-policies",
            page: "tool-policies",
            label: "shell.nav.items.toolPolicies",
            icon: <ShieldCheck {...ICON} />,
            roles: rolesWithCapability("viewToolPolicies"),
          },
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
        route: "usage",
        icon: <BarChart3 {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: "shell.nav.items.usage",
      },
      {
        key: "alerts",
        page: "alerts",
        label: "shell.nav.items.alerts",
        icon: <Bell {...ICON} />,
        roles: rolesWithCapability("viewGlobalSpend"),
      },
      {
        key: "cost-optimization",
        page: "cost-optimization",
        icon: <PiggyBank {...ICON} />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: "shell.nav.items.costOptimization",
      },
      { key: "logs", page: "logs", label: "shell.nav.items.logs", icon: <Activity {...ICON} /> },
      {
        key: "lens",
        page: "lens",
        label: (
          <span className="flex items-center gap-2">
            <Trans i18nKey="shell.nav.items.lens" /> <BetaBadge />
          </span>
        ),
        icon: <Aperture {...ICON} />,
      },
      {
        key: "guardrails-monitor",
        page: "guardrails-monitor",
        label: "shell.nav.items.guardrailsMonitor",
        icon: <HeartPulse {...ICON} />,
        roles: rolesWithCapability("viewGuardrailUsage"),
      },
    ],
  },
  {
    groupLabel: "ACCESS CONTROL",
    items: [
      { key: "teams", page: "teams", label: "shell.nav.items.teams", icon: <Users {...ICON} /> },
      {
        key: "projects",
        page: "projects",
        label: (
          <span className="flex items-center gap-2">
            <Trans i18nKey="shell.nav.items.projects" /> <BetaBadge />
          </span>
        ),
        icon: <Folder {...ICON} />,
        roles: [...projectReaderRoles],
      },
      {
        key: "users",
        page: "users",
        label: "shell.nav.items.internalUsers",
        icon: <User {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "organizations",
        page: "organizations",
        label: "shell.nav.items.organizations",
        icon: <Building2 {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "access-groups",
        page: "access-groups",
        label: "shell.nav.items.accessGroups",
        icon: <Boxes {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "budgets",
        page: "budgets",
        label: "shell.nav.items.budgets",
        icon: <Wallet {...ICON} />,
        roles: all_admin_roles,
      },
    ],
  },
  {
    groupLabel: "DEVELOPER TOOLS",
    items: [
      {
        key: "model-hub-table",
        page: "model-hub-table",
        label: "shell.nav.items.aiHub",
        icon: <LayoutGrid {...ICON} />,
      },
      {
        key: "caching",
        page: "caching",
        label: "shell.nav.items.responseCache",
        icon: <Database {...ICON} />,
        roles: all_admin_roles,
      },
      {
        key: "experimental",
        page: "experimental",
        label: "shell.nav.items.experimental",
        icon: <FlaskConical {...ICON} />,
        children: [
          {
            key: "prompts",
            page: "prompts",
            label: "shell.nav.items.prompts",
            icon: <FileText {...ICON} />,
            roles: rolesWithCapability("viewPrompts"),
          },
          {
            key: "transform-request",
            page: "transform-request",
            label: "shell.nav.items.apiPlayground",
            icon: <Terminal {...ICON} />,
            roles: [...all_admin_roles, ...internalUserRoles],
          },
          {
            key: "tag-management",
            page: "tag-management",
            label: "shell.nav.items.tagManagement",
            icon: <Tags {...ICON} />,
            roles: all_admin_roles,
          },
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
        label: "shell.nav.items.settings",
        icon: <SettingsIcon {...ICON} />,
        roles: all_admin_roles,
        children: [
          {
            key: "router-settings",
            page: "router-settings",
            label: "shell.nav.items.routerSettings",
            icon: <Route {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "logging-and-alerts",
            page: "logging-and-alerts",
            label: "shell.nav.items.loggingAndAlerts",
            icon: <Bell {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "admin-panel",
            page: "admin-panel",
            label: "shell.nav.items.adminSettings",
            icon: <SettingsIcon {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "cost-tracking",
            page: "cost-tracking",
            label: "shell.nav.items.costTracking",
            icon: <BarChart3 {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "ui-theme",
            page: "ui-theme",
            label: "shell.nav.items.uiTheme",
            icon: <Palette {...ICON} />,
            roles: all_admin_roles,
          },
          {
            key: "telemetry",
            page: "telemetry",
            label: "shell.nav.items.telemetry",
            icon: <Activity {...ICON} />,
            roles: proxyAdminTierRoles,
          },
        ],
      },
    ],
  },
];

const HOME_ROUTE = "home";

export const routeOf = (item: MenuItem): string => item.route ?? item.page;

const routeForPathname = (pathname: string): string => routeSegmentForPathname(pathname) || HOME_ROUTE;

const findParentKey = (route: string): string | null => {
  for (const group of menuGroups) {
    for (const item of group.items) {
      if (item.children?.some((c) => routeOf(c) === route)) return item.key;
    }
  }
  return null;
};

const findMenuItemKey = (route: string): string => {
  for (const group of menuGroups) {
    for (const item of group.items) {
      if (routeOf(item) === route) return item.key;
      const child = item.children?.find((c) => routeOf(c) === route);
      if (child) return child.key;
    }
  }
  return HOME_ROUTE;
};

const GROUP_LABELS: Record<string, ParseKeys> = {
  "AI GATEWAY": "shell.nav.groups.aiGateway",
  "CUSTOMER PORTAL": "shell.nav.groups.customerPortal",
  OBSERVABILITY: "shell.nav.groups.observability",
  "ACCESS CONTROL": "shell.nav.groups.accessControl",
  "DEVELOPER TOOLS": "shell.nav.groups.developerTools",
  SETTINGS: "shell.nav.groups.settings",
};

export const groupText = (groupLabel: string): string =>
  Object.hasOwn(GROUP_LABELS, groupLabel) ? i18n.t(GROUP_LABELS[groupLabel]) : groupLabel;

const SECTION_DISPLAY: Record<string, ParseKeys> = {
  "AI GATEWAY": "shell.nav.sections.aiGateway",
  "CUSTOMER PORTAL": "shell.nav.sections.customerPortal",
  OBSERVABILITY: "shell.nav.sections.observability",
  "ACCESS CONTROL": "shell.nav.sections.accessControl",
  "DEVELOPER TOOLS": "shell.nav.sections.developerTools",
  SETTINGS: "shell.nav.sections.settings",
};

export const sectionText = (groupLabel: string): string =>
  Object.hasOwn(SECTION_DISPLAY, groupLabel) ? i18n.t(SECTION_DISPLAY[groupLabel]) : groupLabel;

const prettify = (key: string): string =>
  key
    .split(/[-_]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

export const labelText = (item: MenuItem): string =>
  typeof item.label === "string" ? i18n.t(item.label as ParseKeys) : prettify(item.key);

// Breadcrumb ("Section" / "Page") for the top bar, derived from the same nav config.
export const getBreadcrumb = (pathname: string): { section: string | null; title: string } => {
  const route = routeForPathname(pathname);
  if (route === "aria") return { section: null, title: "Aria Agent" };
  for (const group of [...menuGroups, ...customerPortalGroups]) {
    for (const item of group.items) {
      const section = sectionText(group.groupLabel);
      if (routeOf(item) === route) return { section, title: labelText(item) };
      const child = item.children?.find((c) => routeOf(c) === route);
      if (child) return { section, title: labelText(child) };
    }
  }
  return { section: null, title: prettify(route) };
};

const Sidebar_: React.FC<SidebarProps> = ({
  collapsed = false,
  onToggleCollapsed,
  enabledPagesInternalUsers,
  enableProjectsUI,
  disableAgentsForInternalUsers,
  allowAgentsForTeamAdmins,
  disableVectorStoresForInternalUsers,
  allowVectorStoresForTeamAdmins,
}) => {
  const { userId, accessToken, userRole, isViewOnly } = useAuthorized();
  const { mode: pluginMode } = usePluginMode();
  const { t } = useTranslation();
  const term = useTerminology();
  const labelOf = (item: MenuItem) => (typeof item.label === "string" ? term(labelText(item)) : item.label);
  const isOrgAdmin = useIsOrgAdmin();
  const { data: teams } = useTeams();
  const logout = useLogout(accessToken);

  const currentRoute = routeForPathname(usePathname());
  const selectedKey = currentRoute === "aria" ? null : findMenuItemKey(currentRoute);

  const [openGroups, setOpenGroups] = useState<Set<string>>(() => {
    const parent = findParentKey(currentRoute);
    return new Set(parent ? [parent] : []);
  });

  // Keep the active page's parent group expanded as the user navigates, using the
  // "adjust state during render" pattern rather than an effect (avoids a
  // setState-in-effect render cascade).
  const [prevRoute, setPrevRoute] = useState(currentRoute);
  if (currentRoute !== prevRoute) {
    setPrevRoute(currentRoute);
    const parent = findParentKey(currentRoute);
    if (parent && !openGroups.has(parent)) {
      setOpenGroups((prev) => new Set(prev).add(parent));
    }
  }

  const isTeamAdmin = useMemo(() => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""), [teams, userId]);

  const visibleGroups = useMemo(() => {
    const context: MenuVisibilityContext = {
      userRole,
      isViewOnly,
      isOrgAdmin,
      isTeamAdmin,
      enabledPagesInternalUsers,
      enableProjectsUI,
      disableAgentsForInternalUsers,
      allowAgentsForTeamAdmins,
      disableVectorStoresForInternalUsers,
      allowVectorStoresForTeamAdmins,
    };
    return visibleMenuGroups(pluginMode === "customer-portal" ? customerPortalGroups : menuGroups, context);
  }, [
    pluginMode,
    allowAgentsForTeamAdmins,
    allowVectorStoresForTeamAdmins,
    disableAgentsForInternalUsers,
    disableVectorStoresForInternalUsers,
    enableProjectsUI,
    enabledPagesInternalUsers,
    isOrgAdmin,
    isTeamAdmin,
    isViewOnly,
    userRole,
  ]);

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

  const renderLeaf = (item: MenuItem, isChild: boolean) => {
    const active = selectedKey === item.key;
    const size = isChild ? "sub" : "default";
    const label = <span className="flex-1 truncate group-data-[collapsed=true]/sidebar:hidden">{labelOf(item)}</span>;

    if (item.external_url) {
      return (
        <a
          key={item.key}
          href={item.external_url}
          target="_blank"
          rel="noopener noreferrer"
          title={collapsed ? term(labelText(item)) : undefined}
          data-active={active || undefined}
          className={cn(sidebarMenuButtonVariants({ isActive: active, size }))}
        >
          {item.icon}
          {label}
          <ExternalLink className="size-3.5 shrink-0 opacity-70 group-data-[collapsed=true]/sidebar:hidden" />
        </a>
      );
    }

    return (
      <Link
        key={item.key}
        href={uiHref(routeOf(item))}
        title={collapsed ? term(labelText(item)) : undefined}
        data-active={active || undefined}
        className={cn(sidebarMenuButtonVariants({ isActive: active, size }))}
      >
        {item.icon}
        {label}
      </Link>
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
          aria-expanded={open}
          onClick={() => toggleGroup(item.key)}
          title={collapsed ? term(labelText(item)) : undefined}
        >
          {item.icon}
          <span className="flex-1 truncate group-data-[collapsed=true]/sidebar:hidden">{labelOf(item)}</span>
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

  return (
    <Sidebar collapsed={collapsed}>
      <SidebarBrand
        accessToken={accessToken}
        homeLabel={t("shell.nav.homeLink")}
        collapsed={collapsed}
        onToggleCollapsed={onToggleCollapsed}
      />

      <div className="space-y-1 px-3 pt-3 pb-1">
        <ViewSwitcher className="h-9 w-full max-w-none justify-start shadow-none group-data-[collapsed=true]/sidebar:size-9 group-data-[collapsed=true]/sidebar:justify-center group-data-[collapsed=true]/sidebar:p-0 [&_[data-slot=workspace-label]]:flex-1 [&_[data-slot=workspace-label]]:text-left group-data-[collapsed=true]/sidebar:[&_[data-slot=workspace-label]]:hidden" />
        <AriaSidebarLink active={currentRoute === "aria"} />
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <nav className="flex flex-col gap-0.5 px-3 pb-3">
          {visibleGroups.map((group, gi) => (
            <SidebarGroup key={group.groupLabel}>
              {gi > 0 && <SidebarSeparator className="hidden group-data-[collapsed=true]/sidebar:block" />}
              <SidebarGroupLabel>{term(groupText(group.groupLabel))}</SidebarGroupLabel>
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
