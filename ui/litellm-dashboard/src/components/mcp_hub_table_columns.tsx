import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Info } from "lucide-react";
import { cn } from "@/lib/utils";

export interface MCPServerData {
  server_id: string;
  server_name: string;
  alias?: string | null;
  description?: string | null;
  url: string;
  transport: string;
  auth_type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  credentials?: any;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  teams: string[];
  mcp_access_groups: string[];
  allowed_tools: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extra_headers: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mcp_info: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static_headers: Record<string, any>;
  status: string;
  last_health_check?: string | null;
  health_check_error?: string | null;
  command?: string | null;
  args: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  env: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

function CopyIconButton({
  value,
  onCopy,
  title,
}: {
  value: string;
  onCopy: (text: string) => void;
  title: string;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => onCopy(value)}
            className="cursor-pointer text-muted-foreground hover:text-primary"
            aria-label={title}
          >
            <Copy size={12} />
          </button>
        </TooltipTrigger>
        <TooltipContent>{title}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Categorical status palette. Kept inline (not semantic-token-only) per the
// DEVIATIONS.md categorical-palette policy.
const statusClassMap: Record<string, string> = {
  active:
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300",
  healthy:
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300",
  inactive: "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300",
  unhealthy: "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300",
  unknown: "bg-muted text-muted-foreground",
};

export const mcpHubColumns = (
  showModal: (server: MCPServerData) => void,
  copyToClipboard: (text: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  publicPage: boolean = false,
): ColumnDef<MCPServerData>[] => {
  return [
    {
      header: "Server Name",
      accessorKey: "server_name",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;
        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="font-medium text-sm">{server.server_name}</span>
              <CopyIconButton
                value={server.server_name}
                onCopy={copyToClipboard}
                title="Copy server name"
              />
            </div>
            <div className="md:hidden">
              <span className="text-xs text-muted-foreground">
                {server.description || "-"}
              </span>
            </div>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;
        return (
          <span className="text-xs line-clamp-2">
            {server.description || "-"}
          </span>
        );
      },
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "URL",
      accessorKey: "url",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;
        return (
          <div className="flex items-center space-x-2">
            <span className="text-xs truncate max-w-xs">{server.url}</span>
            <CopyIconButton
              value={server.url}
              onCopy={copyToClipboard}
              title="Copy URL"
            />
          </div>
        );
      },
      meta: { className: "hidden lg:table-cell" },
    },
    {
      header: "Transport",
      accessorKey: "transport",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => (
        <Badge variant="default" className="text-xs">
          {row.original.transport}
        </Badge>
      ),
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "Auth Type",
      accessorKey: "auth_type",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;
        return (
          <Badge
            variant={server.auth_type === "none" ? "outline" : "default"}
            className="text-xs"
          >
            {server.auth_type}
          </Badge>
        );
      },
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "Status",
      accessorKey: "status",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const server = row.original;
        const cls = statusClassMap[server.status] ?? statusClassMap.unknown;
        return (
          <span
            className={cn(
              "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium",
              cls,
            )}
          >
            {server.status || "unknown"}
          </span>
        );
      },
    },
    {
      header: "Tools",
      accessorKey: "allowed_tools",
      enableSorting: false,
      cell: ({ row }) => {
        const server = row.original;
        const tools = server.allowed_tools || [];
        return (
          <div className="space-y-1">
            <span className="text-xs font-medium">
              {tools.length > 0
                ? `${tools.length} tool${tools.length !== 1 ? "s" : ""}`
                : "All tools"}
            </span>
            {tools.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {tools.slice(0, 2).map((tool, idx) => (
                  <Badge
                    key={idx}
                    variant="outline"
                    className="text-xs bg-purple-100 text-purple-800 dark:bg-purple-950/40 dark:text-purple-300 border-purple-200 dark:border-purple-800"
                  >
                    {tool}
                  </Badge>
                ))}
                {tools.length > 2 && (
                  <span className="text-xs text-muted-foreground">
                    +{tools.length - 2}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      },
      meta: { className: "hidden lg:table-cell" },
    },
    {
      header: "Created By",
      accessorKey: "created_by",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => (
        <span className="text-xs">{row.original.created_by || "-"}</span>
      ),
      meta: { className: "hidden xl:table-cell" },
    },
    {
      header: "Public",
      accessorKey: "mcp_info.is_public",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const publicA = rowA.original.mcp_info?.is_public === true ? 1 : 0;
        const publicB = rowB.original.mcp_info?.is_public === true ? 1 : 0;
        return publicA - publicB;
      },
      cell: ({ row }) => {
        const server = row.original;
        return server.mcp_info?.is_public === true ? (
          <Badge variant="default" className="text-xs">
            Yes
          </Badge>
        ) : (
          <Badge variant="outline" className="text-xs">
            No
          </Badge>
        );
      },
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "Details",
      id: "details",
      enableSorting: false,
      cell: ({ row }) => (
        <Button
          size="sm"
          variant="secondary"
          onClick={() => showModal(row.original)}
        >
          <Info className="h-3 w-3" />
          <span className="hidden lg:inline">Details</span>
          <span className="lg:hidden">Info</span>
        </Button>
      ),
    },
  ];
};
