import { useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { MCPServer } from "./types";
import { Check, Pencil, Trash2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { getMaskedAndFullUrl } from "./utils";

const HealthStatusBadge: React.FC<{
  server: MCPServer;
  isLoadingHealth?: boolean;
  isRechecking?: boolean;
  onRecheck?: (serverId: string) => void;
}> = ({ server, isLoadingHealth, isRechecking, onRecheck }) => {
  const [isHovered, setIsHovered] = useState(false);
  const status = server.status || "unknown";
  const lastCheck = server.last_health_check;
  const error = server.health_check_error;

  if (isLoadingHealth || isRechecking) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-muted border border-border">
        <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-pulse"></span>
        Checking
      </span>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "text-emerald-700 bg-emerald-50 border border-emerald-200 dark:text-emerald-300 dark:bg-emerald-950/30 dark:border-emerald-900";
      case "unhealthy":
        return "text-red-700 bg-red-50 border border-red-200 dark:text-red-300 dark:bg-red-950/30 dark:border-red-900";
      default:
        return "text-muted-foreground bg-muted border border-border";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return "✓";
      case "unhealthy":
        return "✗";
      default:
        return "?";
    }
  };

  const isClickable = !!onRecheck;

  const tooltipContent = (
    <div className="max-w-xs">
      <div className="font-semibold mb-1">Health Status: {status}</div>
      {lastCheck && (
        <div className="text-xs mb-1">
          Last Check: {new Date(lastCheck).toLocaleString()}
        </div>
      )}
      {error && (
        <div className="text-xs">
          <div className="font-medium text-red-300 mb-1">Error:</div>
          <div className="break-words">{error}</div>
        </div>
      )}
      {!lastCheck && !error && (
        <div className="text-xs text-muted-foreground">
          No health check data available
        </div>
      )}
      {isClickable && (
        <div className="text-xs text-muted-foreground mt-1">Click to recheck</div>
      )}
    </div>
  );

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full",
              getStatusColor(status),
              isClickable ? "cursor-pointer hover:opacity-80" : "cursor-default",
            )}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={isClickable ? () => onRecheck(server.server_id) : undefined}
          >
            <span>
              {isHovered && isClickable ? "↻" : getStatusIcon(status)}
            </span>
            {isHovered && isClickable
              ? "Recheck"
              : status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        </TooltipTrigger>
        <TooltipContent>{tooltipContent}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export const mcpServerColumns = (
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userRole: string,
  onView: (serverId: string) => void,
  onEdit: (serverId: string) => void,
  onDelete: (serverId: string) => void,
  isLoadingHealth?: boolean,
  onByokConnect?: (server: MCPServer) => void,
  onRecheckHealth?: (serverId: string) => void,
  recheckingServerIds?: Set<string>,
): ColumnDef<MCPServer>[] => [
  {
    accessorKey: "server_id",
    header: "Server ID",
    enableSorting: true,
    cell: ({ row }) => (
      <button
        onClick={() => onView(row.original.server_id)}
        className="font-mono text-blue-600 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 dark:text-blue-300 text-xs font-medium px-2 py-0.5 rounded-md border border-blue-200 dark:border-blue-900 text-left truncate whitespace-nowrap cursor-pointer max-w-[15ch] transition-colors"
      >
        {row.original.server_id.slice(0, 7)}...
      </button>
    ),
  },
  {
    accessorKey: "server_name",
    header: "Name",
    enableSorting: true,
    cell: ({ row }) => {
      const logoUrl = row.original.mcp_info?.logo_url;
      const name = row.original.server_name;
      return (
        <div className="flex items-center gap-2">
          {logoUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={logoUrl}
              alt={`${name ?? "MCP"} logo`}
              className="h-5 w-5 rounded object-contain flex-shrink-0"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
          ) : null}
          <span>{name}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "alias",
    header: "Alias",
    enableSorting: true,
  },
  {
    id: "url",
    header: "URL",
    cell: ({ row }) => {
      const url = row.original.url;
      if (!url) {
        return <span className="text-muted-foreground">—</span>;
      }
      const { maskedUrl } = getMaskedAndFullUrl(url);
      return <span className="font-mono text-sm">{maskedUrl}</span>;
    },
  },
  {
    accessorKey: "transport",
    header: "Transport",
    enableSorting: true,
    cell: ({ row }) => {
      const transport = row.original.transport || "http";
      const specPath = row.original.spec_path;
      const displayTransport =
        specPath && transport !== "stdio" ? "OPENAPI" : transport;
      const label = displayTransport.toUpperCase();
      return (
        <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border bg-muted text-muted-foreground border-border">
          {label}
        </span>
      );
    },
  },
  {
    accessorKey: "auth_type",
    header: "Auth Type",
    enableSorting: true,
    cell: ({ getValue }) => {
      const authType = (getValue() as string) || "none";
      return (
        <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border bg-muted text-muted-foreground border-border">
          {authType}
        </span>
      );
    },
  },
  {
    id: "health_status",
    header: "Health Status",
    cell: ({ row }) => (
      <HealthStatusBadge
        server={row.original}
        isLoadingHealth={isLoadingHealth}
        isRechecking={recheckingServerIds?.has(row.original.server_id)}
        onRecheck={onRecheckHealth}
      />
    ),
  },
  {
    id: "mcp_access_groups",
    header: "Access Groups",
    cell: ({ row }) => {
      const groups = row.original.mcp_access_groups;
      if (Array.isArray(groups) && groups.length > 0) {
        if (typeof groups[0] === "string") {
          const joined = groups.join(", ");
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1 max-w-[200px]">
                    <span className="inline-flex items-center text-xs font-medium px-1.5 py-0.5 rounded bg-muted text-foreground border border-border truncate max-w-[140px]">
                      {groups[0]}
                    </span>
                    {groups.length > 1 && (
                      <span className="text-xs text-muted-foreground font-medium">
                        +{groups.length - 1}
                      </span>
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent>{joined}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        }
      }
      return <span className="text-xs text-muted-foreground">—</span>;
    },
  },
  {
    id: "available_on_public_internet",
    header: "Network Access",
    cell: ({ row }) => {
      const isPublic = row.original.available_on_public_internet;
      return isPublic ? (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-900 text-xs font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
          Public
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-orange-50 text-orange-700 dark:bg-orange-950/30 dark:text-orange-300 rounded-full border border-orange-200 dark:border-orange-900 text-xs font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-orange-500"></span>
          Internal
        </span>
      );
    },
  },
  {
    header: "Created",
    accessorKey: "created_at",
    enableSorting: true,
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.created_at)
        return <span className="text-xs text-muted-foreground">—</span>;
      const date = new Date(server.created_at);
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs text-muted-foreground">
                {date.toLocaleDateString()}
              </span>
            </TooltipTrigger>
            <TooltipContent>{date.toLocaleString()}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
  },
  {
    header: "Updated",
    accessorKey: "updated_at",
    enableSorting: true,
    sortingFn: "datetime",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.updated_at)
        return <span className="text-xs text-muted-foreground">—</span>;
      const date = new Date(server.updated_at);
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs text-muted-foreground">
                {date.toLocaleDateString()}
              </span>
            </TooltipTrigger>
            <TooltipContent>{date.toLocaleString()}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
  },
  {
    id: "byok_credential",
    header: "Credential",
    cell: ({ row }) => {
      const server = row.original;
      if (!server.is_byok) {
        return <span className="text-muted-foreground text-xs">—</span>;
      }
      if (server.has_user_credential) {
        return (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-950/30 dark:text-emerald-300 dark:border-emerald-900">
              <Check className="h-2.5 w-2.5" /> Connected
            </span>
            {onByokConnect && (
              <button
                className="text-xs text-muted-foreground hover:text-primary transition-colors"
                onClick={() => onByokConnect(server)}
              >
                Update
              </button>
            )}
          </div>
        );
      }
      return onByokConnect ? (
        <button
          className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-md font-medium transition-colors shadow-sm"
          onClick={() => onByokConnect(server)}
        >
          Connect
        </button>
      ) : null;
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => (
      <div className="flex items-center gap-1">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onEdit(row.original.server_id)}
                className="p-1.5 rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                aria-label="Edit"
              >
                <Pencil className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Edit</TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => onDelete(row.original.server_id)}
                className="p-1.5 rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                aria-label="Delete"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Delete</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    ),
  },
];
