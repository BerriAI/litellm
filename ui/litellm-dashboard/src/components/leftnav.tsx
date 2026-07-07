"use client";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  Key,
  PlayCircle,
  MessageSquare,
  Blocks,
  Bot,
  GitBranch,
  BookOpen,
  Wrench,
  Search,
  Database,
  Shield,
  ClipboardCheck,
  Code,
  BarChart3,
  LineChart,
  Users,
  Folder,
  User,
  Building2,
  CreditCard,
  LayoutGrid,
  FlaskConical,
  FileText,
  Tags,
  Settings,
  Palette,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { useMemo, useState } from "react";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/cva.config";

interface SidebarProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  collapsed?: boolean;
  enabledPagesInternalUsers?: string[] | null;
  enableProjectsUI?: boolean;
  enableChatUI?: boolean;
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
      {
        key: "api-keys",
        page: "api-keys",
        label: "Virtual Keys",
        icon: <Key className="size-4" />,
      },
      {
        key: "llm-playground",
        page: "llm-playground",
        label: "Playground",
        icon: <PlayCircle className="size-4" />,
        roles: rolesWithWriteAccess,
      },
      {
        key: "chat",
        page: "chat",
        label: (
          <span className="flex items-center gap-2">
            Chat <NewBadge />
          </span>
        ),
        icon: <MessageSquare className="size-4" />,
      },
      {
        key: "models",
        page: "models",
        label: "Models + Endpoints",
        icon: <Blocks className="size-4" />,
        roles: rolesAllowedToViewWriteScopedPages,
      },
      {
        key: "agentic",
        page: "agentic",
        label: "Agentic",
        icon: <Bot className="size-4" />,
        children: [
          {
            key: "agents",
            page: "agents",
            label: "Agents",
            icon: <Bot className="size-4" />,
            roles: rolesAllowedToViewWriteScopedPages,
          },
          {
            key: "workflows",
            page: "workflows",
            label: "Workflow Runs",
            icon: <GitBranch className="size-4" />,
          },
          {
            key: "memory",
            page: "memory",
            label: "Memory",
            icon: <BookOpen className="size-4" />,
          },
        ],
      },
      {
        key: "mcp-servers",
        page: "mcp-servers",
        label: "MCP Servers",
        icon: <Wrench className="size-4" />,
      },
      {
        key: "skills",
        page: "skills",
        label: "Skills",
        icon: <Code className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "guardrails",
        page: "guardrails",
        label: "Guardrails",
        icon: <Shield className="size-4" />,
      },
      {
        key: "policies",
        page: "policies",
        label: "Policies",
        icon: <ClipboardCheck className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "tools",
        page: "tools",
        label: "Tools",
        icon: <Wrench className="size-4" />,
        children: [
          {
            key: "search-tools",
            page: "search-tools",
            label: "Search Tools",
            icon: <Search className="size-4" />,
          },
          {
            key: "vector-stores",
            page: "vector-stores",
            label: "Vector Stores",
            icon: <Database className="size-4" />,
          },
          {
            key: "tool-policies",
            page: "tool-policies",
            label: "Tool Policies",
            icon: <Shield className="size-4" />,
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
        icon: <BarChart3 className="size-4" />,
        roles: [...all_admin_roles, ...internalUserRoles],
        label: "Usage",
      },
      {
        key: "logs",
        page: "logs",
        label: "Logs",
        icon: <LineChart className="size-4" />,
      },
      {
        key: "guardrails-monitor",
        page: "guardrails-monitor",
        label: "Guardrails Monitor",
        icon: <Shield className="size-4" />,
        roles: [...all_admin_roles, ...internalUserRoles],
      },
    ],
  },
  {
    groupLabel: "ACCESS CONTROL",
    items: [
      {
        key: "teams",
        page: "teams",
        label: "Teams",
        icon: <Users className="size-4" />,
      },
      {
        key: "projects",
        page: "projects",
        label: (
          <span className="flex items-center gap-2">
            Projects <NewBadge />
          </span>
        ),
        icon: <Folder className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "users",
        page: "users",
        label: "Internal Users",
        icon: <User className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "organizations",
        page: "organizations",
        label: "Organizations",
        icon: <Building2 className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "access-groups",
        page: "access-groups",
        label: "Access Groups",
        icon: <Blocks className="size-4" />,
        roles: all_admin_roles,
      },
      {
        key: "budgets",
        page: "budgets",
        label: "Budgets",
        icon: <CreditCard className="size-4" />,
        roles: all_admin_roles,
      },
    ],
  },
  {
    groupLabel: "DEVELOPER TOOLS",
    items: [
      {
        key: "api_ref",
        page: "api_ref",
        label: "API Reference",
        icon: <Code className="size-4" />,
      },
      {
        key: "model-hub-table",
        page: "model-hub-table",
        label: "AI Hub",
        icon: <LayoutGrid className="size-4" />,
      },
      {
        key: "learning-resources",
        page: "learning-resources",
        label: "Learning Resources",
        icon: <BookOpen className="size-4" />,
        external_url: "https://models.litellm.ai/cookbook",
      },
      {
        key: "experimental",
        page: "experimental",
        label: "Experimental",
        icon: <FlaskConical className="size-4" />,
        children: [
          {
            key: "caching",
            page: "caching",
            label: "Caching",
            icon: <Database className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "prompts",
            page: "prompts",
            label: "Prompts",
            icon: <FileText className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "transform-request",
            page: "transform-request",
            label: "API Playground",
            icon: <Code className="size-4" />,
            roles: [...all_admin_roles, ...internalUserRoles],
          },
          {
            key: "tag-management",
            page: "tag-management",
            label: "Tag Management",
            icon: <Tags className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "4",
            page: "usage",
            label: "Old Usage",
            icon: <BarChart3 className="size-4" />,
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
        label: (
          <span className="flex items-center gap-2">
            Settings <NewBadge />
          </span>
        ),
        icon: <Settings className="size-4" />,
        roles: all_admin_roles,
        children: [
          {
            key: "router-settings",
            page: "router-settings",
            label: "Router Settings",
            icon: <Settings className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "logging-and-alerts",
            page: "logging-and-alerts",
            label: "Logging & Alerts",
            icon: <Settings className="size-4" />,
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
            icon: <Settings className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "cost-tracking",
            page: "cost-tracking",
            label: "Cost Tracking",
            icon: <BarChart3 className="size-4" />,
            roles: all_admin_roles,
          },
          {
            key: "ui-theme",
            page: "ui-theme",
            label: "UI Theme",
            icon: <Palette className="size-4" />,
            roles: all_admin_roles,
          },
        ],
      },
    ],
  },
];

function NavItem({
  item,
  isSelected,
  collapsed,
  onNavigate,
}: {
  item: MenuItem;
  isSelected: boolean;
  collapsed: boolean;
  onNavigate: (page: string) => void;
}) {
  const migratedRoute = MIGRATED_PAGES[item.page];
  const href = item.external_url ?? (migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(item.page));
  const isExternal = Boolean(item.external_url);

  const handleClick = (e: React.MouseEvent) => {
    if (isExternal) {
      return;
    }
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) {
      return;
    }
    e.preventDefault();
    onNavigate(item.page);
  };

  const content = (
    <a
      role="menuitem"
      href={href}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
      onClick={handleClick}
      className={cn(
        "flex items-center gap-2 h-8 py-1.5 px-2 rounded-md text-[13px] transition-colors w-full",
        "hover:bg-sidebar-accent",
        isSelected && "bg-sidebar-accent text-sidebar-accent-foreground font-medium",
        !isSelected && "text-sidebar-foreground/70",
      )}
    >
      <span className="shrink-0">{item.icon}</span>
      {!collapsed && (
        <span className="truncate flex items-center gap-1.5">
          {item.label}
          {isExternal && <ExternalLink aria-hidden className="size-3 opacity-60" />}
        </span>
      )}
    </a>
  );

  if (collapsed) {
    const tooltipLabel = typeof item.label === "string" ? item.label : item.page;
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent side="right">{tooltipLabel}</TooltipContent>
      </Tooltip>
    );
  }

  return content;
}

function CollapsibleNavItem({
  item,
  selectedKey,
  collapsed,
  onNavigate,
}: {
  item: MenuItem;
  selectedKey: string;
  collapsed: boolean;
  onNavigate: (page: string) => void;
}) {
  const hasSelectedChild = item.children?.some((child) => child.key === selectedKey) ?? false;
  const [open, setOpen] = useState(hasSelectedChild);

  if (collapsed) {
    const tooltipLabel = typeof item.label === "string" ? item.label : item.page;
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="flex items-center justify-center h-8 py-1.5 px-2 rounded-md text-[13px] text-sidebar-foreground/70 hover:bg-sidebar-accent transition-colors w-full"
          >
            <span className="shrink-0">{item.icon}</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">{tooltipLabel}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          role="menuitem"
          className={cn(
            "flex items-center gap-2 h-8 py-1.5 px-2 rounded-md text-[13px] transition-colors w-full",
            "hover:bg-sidebar-accent text-sidebar-foreground/70",
          )}
        >
          <span className="shrink-0">{item.icon}</span>
          <span className="truncate flex-1 text-left">{item.label}</span>
          <ChevronRight className={cn("size-3.5 text-muted-foreground transition-transform", open && "rotate-90")} />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="ml-4 border-l border-sidebar-border pl-2 mt-0.5 flex flex-col gap-0.5">
          {item.children?.map((child) => (
            <NavItem
              key={child.key}
              item={child}
              isSelected={child.key === selectedKey}
              collapsed={false}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

const Sidebar: React.FC<SidebarProps> = ({
  setPage,
  defaultSelectedKey,
  collapsed = false,
  enabledPagesInternalUsers,
  enableProjectsUI,
  enableChatUI,
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
      org.members?.some((member) => member.user_id === userId && member.user_role === "org_admin"),
    );
  }, [userId, organizations]);

  const isTeamAdmin = useMemo(() => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""), [teams, userId]);

  const navigateToPage = (page: string) => setPage(page);

  const isFeatureGated = (item: MenuItem): boolean => {
    if (item.key === "projects" && !enableProjectsUI) return true;
    if (item.key === "chat" && !enableChatUI) return true;
    return false;
  };

  const isDisabledForNonAdmin = (item: MenuItem): boolean => {
    if (item.key === "agents" && disableAgentsForInternalUsers && !(allowAgentsForTeamAdmins && isTeamAdmin))
      return true;
    if (
      item.key === "vector-stores" &&
      disableVectorStoresForInternalUsers &&
      !(allowVectorStoresForTeamAdmins && isTeamAdmin)
    )
      return true;
    return false;
  };

  const isVisibleForInternalUser = (item: MenuItem, enabledPages: string[]): boolean => {
    if (item.children && item.children.length > 0) {
      return item.children.some((child) => enabledPages.includes(child.page));
    }
    return enabledPages.includes(item.page);
  };

  const filterItemsByRole = (items: MenuItem[]): MenuItem[] => {
    const isAdmin = isAdminRole(userRole);
    const hasPageRestrictions = !isAdmin && enabledPagesInternalUsers != null;

    return items
      .map((item) => ({
        ...item,
        children: item.children ? filterItemsByRole(item.children) : undefined,
      }))
      .filter((item) => {
        if (item.key === "organizations" || item.key === "users") {
          const hasRoleAccess = !item.roles || item.roles.includes(userRole) || isOrgAdmin;
          if (!hasRoleAccess) return false;
          if (hasPageRestrictions) return enabledPagesInternalUsers!.includes(item.page);
          return true;
        }

        if (isFeatureGated(item)) return false;
        if (!isAdmin && isDisabledForNonAdmin(item)) return false;
        if (item.roles && !item.roles.includes(userRole)) return false;
        if (hasPageRestrictions) return isVisibleForInternalUser(item, enabledPagesInternalUsers!);
        return true;
      });
  };

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
    <TooltipProvider>
      <aside
        className={cn(
          "bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300",
          collapsed ? "w-[52px]" : "w-[220px]",
        )}
      >
        <ScrollArea className="flex-1">
          <nav className="flex flex-col gap-4 py-2 px-2">
            {menuGroups
              .filter((group) => !group.roles || group.roles.includes(userRole))
              .map((group) => {
                const filteredItems = filterItemsByRole(group.items);
                if (filteredItems.length === 0) return null;

                return (
                  <div key={group.groupLabel} className="flex flex-col gap-0.5">
                    {!collapsed && (
                      <span className="text-[10px] font-semibold tracking-[0.05em] text-muted-foreground px-2 pt-1 pb-0.5">
                        {group.groupLabel}
                      </span>
                    )}
                    {filteredItems.map((item) =>
                      item.children && item.children.length > 0 ? (
                        <CollapsibleNavItem
                          key={item.key}
                          item={item}
                          selectedKey={selectedMenuKey}
                          collapsed={collapsed}
                          onNavigate={navigateToPage}
                        />
                      ) : (
                        <NavItem
                          key={item.key}
                          item={item}
                          isSelected={item.key === selectedMenuKey}
                          collapsed={collapsed}
                          onNavigate={navigateToPage}
                        />
                      ),
                    )}
                  </div>
                );
              })}
          </nav>
        </ScrollArea>
        {isAdminRole(userRole) && !collapsed && <UsageIndicator accessToken={accessToken} width={220} />}
      </aside>
    </TooltipProvider>
  );
};

export default Sidebar;

export { menuGroups };
