import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Link as LinkIcon } from "lucide-react";
import { Plugin } from "./claude_code_plugins/types";

export const skillHubColumns = (
  showModal: (skill: Plugin) => void,
  copyToClipboard: (text: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  publicPage: boolean = false,
): ColumnDef<Plugin>[] => [
  {
    header: "Skill Name",
    accessorKey: "name",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const skill = row.original;
      return (
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <button
              type="button"
              className="font-medium text-sm cursor-pointer text-primary hover:underline bg-transparent border-none p-0"
              onClick={() => showModal(skill)}
            >
              {skill.name}
            </button>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(skill.name)}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy skill name"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy skill name</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          {skill.description && (
            <p className="text-xs text-muted-foreground line-clamp-1 md:hidden">
              {skill.description}
            </p>
          )}
        </div>
      );
    },
  },
  {
    header: "Description",
    accessorKey: "description",
    enableSorting: false,
    cell: ({ row }) => (
      <p className="text-xs line-clamp-2">
        {row.original.description || "-"}
      </p>
    ),
  },
  {
    header: "Category",
    accessorKey: "category",
    enableSorting: true,
    cell: ({ row }) => {
      const cat = row.original.category;
      if (!cat) return <span className="text-xs text-muted-foreground">-</span>;
      return (
        <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 text-xs">
          {cat}
        </Badge>
      );
    },
  },
  {
    header: "Domain",
    accessorKey: "domain",
    enableSorting: true,
    cell: ({ row }) => (
      <span className="text-xs">{row.original.domain || "-"}</span>
    ),
  },
  {
    header: "Source",
    accessorKey: "source",
    enableSorting: false,
    cell: ({ row }) => {
      const src = row.original.source;
      let url: string | null = null;
      let label = "-";
      if (src?.source === "github" && src.repo) {
        url = `https://github.com/${src.repo}`;
        label = src.repo;
      } else if (src?.source === "git-subdir" && src.url) {
        url = src.path ? `${src.url}/tree/main/${src.path}` : src.url;
        label = url.replace("https://github.com/", "");
      } else if (src?.source === "url" && src.url) {
        url = src.url;
        label = src.url.replace(/^https?:\/\//, "");
      }
      if (!url)
        return <span className="text-xs text-muted-foreground">-</span>;
      return (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-primary hover:underline truncate max-w-[180px]"
          title={label}
        >
          <span className="truncate">{label}</span>
          <LinkIcon className="h-2.5 w-2.5 shrink-0" />
        </a>
      );
    },
  },
  {
    header: "Status",
    accessorKey: "enabled",
    enableSorting: true,
    cell: ({ row }) => (
      <Badge
        className={
          row.original.enabled
            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300 text-xs"
            : "bg-muted text-muted-foreground text-xs"
        }
      >
        {row.original.enabled ? "Public" : "Draft"}
      </Badge>
    ),
  },
];
