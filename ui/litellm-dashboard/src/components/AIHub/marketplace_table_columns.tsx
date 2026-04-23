import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Copy } from "lucide-react";
import { MarketplacePluginEntry } from "@/components/claude_code_plugins/types";
import {
  formatInstallCommand,
  getCategoryBadgeColor,
  getSourceDisplayText,
} from "@/components/claude_code_plugins/helpers";

// Map tremor palette names to categorical Tailwind classes
const categoryColorClass: Record<string, string> = {
  blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  green:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  emerald:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  amber:
    "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  yellow:
    "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  red: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  purple:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  indigo:
    "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300",
  gray: "bg-muted text-muted-foreground",
};

export const getMarketplaceTableColumns = (
  copyToClipboard: (text: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  publicPage: boolean = false,
): ColumnDef<MarketplacePluginEntry>[] => [
  {
    header: "Plugin Name",
    accessorKey: "name",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const plugin = row.original;
      const installCommand = formatInstallCommand(plugin);

      return (
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="font-medium text-sm">{plugin.name}</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(installCommand)}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy install command"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy install command</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="md:hidden">
            <p className="text-xs text-muted-foreground">
              {plugin.description || "No description"}
            </p>
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
      const plugin = row.original;
      return (
        <p className="text-xs line-clamp-2">{plugin.description || "-"}</p>
      );
    },
    meta: {
      className: "hidden md:table-cell",
    },
  },
  {
    header: "Version",
    accessorKey: "version",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const plugin = row.original;
      return plugin.version ? (
        <Badge className={cn("text-xs", categoryColorClass.blue)}>
          v{plugin.version}
        </Badge>
      ) : (
        <span className="text-xs text-muted-foreground">-</span>
      );
    },
    meta: {
      className: "hidden lg:table-cell",
    },
  },
  {
    header: "Category",
    accessorKey: "category",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const plugin = row.original;
      const badgeColor = getCategoryBadgeColor(plugin.category);

      return plugin.category ? (
        <Badge
          className={cn(
            "text-xs",
            categoryColorClass[badgeColor] || categoryColorClass.gray,
          )}
        >
          {plugin.category}
        </Badge>
      ) : (
        <Badge className={cn("text-xs", categoryColorClass.gray)}>
          Uncategorized
        </Badge>
      );
    },
    meta: {
      className: "hidden lg:table-cell",
    },
  },
  {
    header: "Source",
    accessorKey: "source",
    enableSorting: false,
    cell: ({ row }) => {
      const plugin = row.original;
      return (
        <span className="text-xs text-muted-foreground">
          {getSourceDisplayText(plugin.source)}
        </span>
      );
    },
    meta: {
      className: "hidden xl:table-cell",
    },
  },
  {
    header: "Keywords",
    accessorKey: "keywords",
    enableSorting: false,
    cell: ({ row }) => {
      const plugin = row.original;
      const keywords = plugin.keywords?.slice(0, 3) || [];
      const remaining = (plugin.keywords?.length || 0) - 3;

      return (
        <div className="flex flex-wrap gap-1">
          {keywords.map((keyword, index) => (
            <Badge
              key={index}
              className={cn("text-xs", categoryColorClass.gray)}
            >
              {keyword}
            </Badge>
          ))}
          {remaining > 0 && (
            <Badge className={cn("text-xs", categoryColorClass.gray)}>
              +{remaining}
            </Badge>
          )}
        </div>
      );
    },
    meta: {
      className: "hidden xl:table-cell",
    },
  },
  {
    header: "Install Command",
    id: "install_command",
    enableSorting: false,
    cell: ({ row }) => {
      const plugin = row.original;
      const installCommand = formatInstallCommand(plugin);

      return (
        <div className="flex items-center space-x-2">
          <code className="text-xs bg-muted px-2 py-1 rounded font-mono truncate max-w-[200px]">
            {installCommand}
          </code>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="secondary"
                  onClick={() => copyToClipboard(installCommand)}
                  className="h-7 w-7"
                  aria-label="Copy command"
                >
                  <Copy className="h-3 w-3" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Copy command</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
];
