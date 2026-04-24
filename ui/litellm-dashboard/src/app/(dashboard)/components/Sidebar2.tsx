"use client";

import {
  Key as KeyOutlined,
  PlayCircle as PlayCircleOutlined,
  Square as BlockOutlined,
  BarChart3 as BarChartOutlined,
  Users as TeamOutlined,
  Landmark as BankOutlined,
  User as UserOutlined,
  Settings as SettingOutlined,
  Plug as ApiOutlined,
  LayoutGrid as AppstoreOutlined,
  Database as DatabaseOutlined,
  FileText as FileTextOutlined,
  LineChart as LineChartOutlined,
  ShieldCheck as SafetyOutlined,
  FlaskConical as ExperimentOutlined,
  Wrench as ToolOutlined,
  Tags as TagsOutlined,
  ClipboardCheck as AuditOutlined,
  ChevronDown,
} from "lucide-react";
import * as React from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  all_admin_roles,
  internalUserRoles,
  isAdminRole,
  rolesWithWriteAccess,
} from "@/utils/roles";
import UsageIndicator from "@/components/UsageIndicator";
import { serverRootPath } from "@/components/networking";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// -------- Types --------
interface SidebarProps {
  accessToken: string | null;
  userRole: string;
  /** Fallback selection id (legacy), used if path can't be matched */
  defaultSelectedKey: string;
  collapsed?: boolean;
}

interface MenuItemCfg {
  key: string;
  newTab?: boolean;
  page: string; // legacy id; we map this to a path below
  label: string;
  roles?: string[];
  children?: MenuItemCfg[];
  icon?: React.ReactNode;
}

/** ---------- Base URL helpers ---------- */
/**
 * Normalizes NEXT_PUBLIC_BASE_URL to either "/" or "/ui/" (always with a trailing slash).
 * Supported env values: "" or "ui/".
 * Also considers the serverRootPath from the proxy config (e.g., "/my-custom-path").
 */
const getBasePath = () => {
  const raw = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = raw.replace(/^\/+|\/+$/g, ""); // strip leading/trailing slashes
  const uiPath = trimmed ? `/${trimmed}/` : "/";

  // If serverRootPath is set and not "/", prepend it to the UI path
  if (serverRootPath && serverRootPath !== "/") {
    // Remove trailing slash from serverRootPath and ensure uiPath has no leading slash for proper joining
    const cleanServerRoot = serverRootPath.replace(/\/+$/, "");
    const cleanUiPath = uiPath.replace(/^\/+/, "");
    return `${cleanServerRoot}/${cleanUiPath}`;
  }

  return uiPath;
};

/** Map legacy `page` ids to real app routes (relative, no leading slash). */
const routeFor = (slug: string): string => {
  switch (slug) {
    // top level
    case "api-keys":
      return "virtual-keys";
    case "llm-playground":
      return "test-key";
    case "models":
      return "models-and-endpoints";
    case "new_usage":
      return "usage";
    case "teams":
      return "teams";
    case "organizations":
      return "organizations";
    case "users":
      return "users";
    case "api_ref":
      return "api-reference";
    case "model-hub-table":
      // If you intend the newer in-dashboard page, use "model-hub".
      return "model-hub";
    case "logs":
      return "logs";
    case "guardrails":
      return "guardrails";
    case "policies":
      return "policies";
    case "chat":
      return "chat";

    // tools
    case "mcp-servers":
      return "tools/mcp-servers";
    case "vector-stores":
      return "tools/vector-stores";
    case "byok-demo":
      return "tools/byok-demo";

    // experimental
    case "caching":
      return "experimental/caching";
    case "prompts":
      return "experimental/prompts";
    case "budgets":
      return "experimental/budgets";
    case "transform-request":
      return "experimental/api-playground";
    case "tag-management":
      return "experimental/tag-management";
    case "claude-code-plugins":
      return "experimental/claude-code-plugins";
    case "usage": // "Old Usage"
      return "experimental/old-usage";

    // settings
    case "general-settings":
      return "settings/router-settings";
    case "settings": // "Logging & Alerts"
      return "settings/logging-and-alerts";
    case "admin-panel":
      return "settings/admin-settings";
    case "ui-theme":
      return "settings/ui-theme";

    default:
      // treat as already a relative path
      return slug.replace(/^\/+/, "");
  }
};

/** Prefix base path ("/" or "/ui/") */
const toHref = (slugOrPath: string) => {
  const base = getBasePath(); // "/" or "/ui/"
  const rel = routeFor(slugOrPath).replace(/^\/+|\/+$/g, "");
  return `${base}${rel}`;
};

// ----- Menu config (unchanged labels/icons; same appearance) -----
const menuItems: MenuItemCfg[] = [
  { key: "1", page: "api-keys", label: "Virtual Keys", icon: <KeyOutlined style={{ fontSize: 18 }} /> },
  {
    key: "3",
    page: "llm-playground",
    label: "Test Key",
    icon: <PlayCircleOutlined style={{ fontSize: 18 }} />,
    roles: rolesWithWriteAccess,
  },
  {
    key: "2",
    page: "models",
    label: "Models + Endpoints",
    icon: <BlockOutlined style={{ fontSize: 18 }} />,
    roles: rolesWithWriteAccess,
  },
  {
    key: "12",
    page: "new_usage",
    label: "Usage",
    icon: <BarChartOutlined style={{ fontSize: 18 }} />,
    roles: [...all_admin_roles, ...internalUserRoles],
  },
  { key: "6", page: "teams", label: "Teams", icon: <TeamOutlined style={{ fontSize: 18 }} /> },
  {
    key: "17",
    page: "organizations",
    label: "Organizations",
    icon: <BankOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
  },
  {
    key: "5",
    page: "users",
    label: "Internal Users",
    icon: <UserOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
  },
  { key: "14", page: "api-reference", label: "API Reference", icon: <ApiOutlined style={{ fontSize: 18 }} /> },
  {
    key: "16",
    page: "model-hub-table",
    label: "Model Hub",
    icon: <AppstoreOutlined style={{ fontSize: 18 }} />,
  },
  { key: "15", page: "logs", label: "Logs", icon: <LineChartOutlined style={{ fontSize: 18 }} /> },
  {
    key: "11",
    page: "guardrails",
    label: "Guardrails",
    icon: <SafetyOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
  },
  {
    key: "28",
    page: "policies",
    label: "Policies",
    icon: <AuditOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
  },
  {
    key: "26",
    page: "tools",
    label: "Tools",
    icon: <ToolOutlined style={{ fontSize: 18 }} />,
    children: [
      { key: "18", page: "mcp-servers", label: "MCP Servers", icon: <ToolOutlined style={{ fontSize: 18 }} /> },
      {
        key: "21",
        page: "vector-stores",
        label: "Vector Stores",
        icon: <DatabaseOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
    ],
  },
  {
    key: "experimental",
    page: "experimental",
    label: "Experimental",
    icon: <ExperimentOutlined style={{ fontSize: 18 }} />,
    children: [
      {
        key: "9",
        page: "caching",
        label: "Caching",
        icon: <DatabaseOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "25",
        page: "prompts",
        label: "Prompts",
        icon: <FileTextOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "10",
        page: "budgets",
        label: "Budgets",
        icon: <BankOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "20",
        page: "transform-request",
        label: "API Playground",
        icon: <ApiOutlined style={{ fontSize: 18 }} />,
        roles: [...all_admin_roles, ...internalUserRoles],
      },
      {
        key: "19",
        page: "tag-management",
        label: "Tag Management",
        icon: <TagsOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "27",
        page: "claude-code-plugins",
        label: "Claude Code Plugins",
        icon: <ToolOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      { key: "4", page: "usage", label: "Old Usage", icon: <BarChartOutlined style={{ fontSize: 18 }} /> },
    ],
  },
  {
    key: "settings",
    page: "settings",
    label: "Settings",
    icon: <SettingOutlined style={{ fontSize: 18 }} />,
    roles: all_admin_roles,
    children: [
      {
        key: "11",
        page: "general-settings",
        label: "Router Settings",
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "8",
        page: "settings",
        label: "Logging & Alerts",
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "13",
        page: "admin-panel",
        label: "Admin Settings",
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
      {
        key: "14",
        page: "ui-theme",
        label: "UI Theme",
        icon: <SettingOutlined style={{ fontSize: 18 }} />,
        roles: all_admin_roles,
      },
    ],
  },
];

const Sidebar2: React.FC<SidebarProps> = ({ accessToken, userRole, defaultSelectedKey, collapsed = false }) => {
  const router = useRouter();
  const pathname = usePathname() || "/";

  // ----- Filter by role without mutating originals -----
  const filteredMenuItems = React.useMemo<MenuItemCfg[]>(() => {
    return menuItems
      .filter((item) => !item.roles || item.roles.includes(userRole))
      .map((item) => ({
        ...item,
        children: item.children ? item.children.filter((c) => !c.roles || c.roles.includes(userRole)) : undefined,
      }));
  }, [userRole]);

  // ----- Compute selected key from current path -----
  const selectedMenuKey = React.useMemo(() => {
    const base = getBasePath();
    // strip base prefix and leading slash -> "virtual-keys", "tools/mcp-servers", etc.
    const rel = pathname.startsWith(base) ? pathname.slice(base.length) : pathname.replace(/^\/+/, "");
    const relLower = rel.toLowerCase();

    const matchesPath = (slug: string) => {
      const route = routeFor(slug).toLowerCase();
      return relLower === route || relLower.startsWith(`${route}/`);
    };

    // search top-level
    for (const item of filteredMenuItems) {
      if (!item.children && matchesPath(item.page)) return item.key;
      if (item.children) {
        for (const child of item.children) {
          if (matchesPath(child.page)) return child.key;
        }
      }
    }

    // fallback to legacy defaultSelectedKey mapping
    const fallback = filteredMenuItems.find((i) => i.page === defaultSelectedKey)?.key;
    if (fallback) return fallback;

    for (const item of filteredMenuItems) {
      if (item.children?.some((c) => c.page === defaultSelectedKey)) {
        const child = item.children.find((c) => c.page === defaultSelectedKey)!;
        return child.key;
      }
    }

    return "1";
  }, [pathname, filteredMenuItems, defaultSelectedKey]);

  // ----- Navigation -----
  const goTo = (slug: string, newTab?: boolean) => {
    const href = toHref(slug);
    if (newTab) {
      window.open(href, "_blank");
    } else {
      router.push(href);
    }
  };

  // Track which submenu parents are open. Defaults to none expanded in
  // collapsed mode (the whole panel collapses). In expanded mode, open any
  // parent whose child is currently selected.
  const initialOpenKeys = React.useMemo(() => {
    const open: string[] = [];
    for (const item of filteredMenuItems) {
      if (item.children?.some((c) => c.key === selectedMenuKey)) {
        open.push(item.key);
      }
    }
    return open;
  }, [filteredMenuItems, selectedMenuKey]);

  const [openKeys, setOpenKeys] = React.useState<string[]>(initialOpenKeys);

  const toggleKey = (key: string) => {
    setOpenKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const renderItem = (item: MenuItemCfg, depth: number) => {
    const isSelected = item.key === selectedMenuKey;
    const hasChildren = !!item.children && item.children.length > 0;
    const isOpen = openKeys.includes(item.key);
    const indent = depth * 12;

    const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
      e.preventDefault();
      if (hasChildren) {
        toggleKey(item.key);
      } else {
        goTo(item.page, item.newTab);
      }
    };

    const href = toHref(item.page);
    const itemClasses = cn(
      "flex items-center gap-2 rounded-md text-sm px-2 mx-1 h-[34px]",
      "transition-colors cursor-pointer text-foreground no-underline",
      isSelected
        ? "bg-primary/10 text-primary font-medium"
        : "hover:bg-muted",
    );

    const linkContent = (
      <>
        {item.icon && (
          <span
            className={cn(
              "inline-flex items-center justify-center shrink-0",
              "h-[18px] w-[18px]",
              "[&>svg]:h-[18px] [&>svg]:w-[18px]",
            )}
          >
            {item.icon}
          </span>
        )}
        {!collapsed && <span className="flex-1 truncate">{item.label}</span>}
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

    const link = (
      <a
        href={href}
        target={item.newTab ? "_blank" : undefined}
        rel={item.newTab ? "noopener noreferrer" : undefined}
        onClick={handleClick}
        style={{ paddingLeft: `${8 + indent}px` }}
        className={itemClasses}
      >
        {linkContent}
      </a>
    );

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
          <ul role="group" className="mt-[2px] mb-[2px] list-none p-0">
            {item.children!.map((child) => renderItem(child, depth + 1))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <aside
      className={cn(
        "relative bg-background flex flex-col min-h-screen",
        "transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]",
      )}
      style={{ width: collapsed ? 80 : 220 }}
    >
      <nav
        aria-label="Main navigation"
        className="custom-sidebar-menu flex-1 overflow-y-auto"
      >
        <ul role="menu" className="list-none p-0 m-0">
          {filteredMenuItems.map((item) => renderItem(item, 0))}
        </ul>
      </nav>
      {isAdminRole(userRole) && !collapsed && (
        <UsageIndicator accessToken={accessToken} width={220} />
      )}
    </aside>
  );
};

export default Sidebar2;
