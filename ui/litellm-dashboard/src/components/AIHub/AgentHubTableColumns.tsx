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

export interface AgentHubData {
  agent_id?: string;
  protocolVersion: string;
  name: string;
  description: string;
  url: string;
  version: string;
  capabilities?: {
    streaming?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
  };
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  skills?: Array<{
    id: string;
    name: string;
    description: string;
    tags?: string[];
    examples?: string[];
  }>;
  supportsAuthenticatedExtendedCard?: boolean;
  is_public?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

const badgeClass = {
  blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  emerald:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  purple:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  gray: "bg-muted text-muted-foreground",
};

export const getAgentHubTableColumns = (
  showModal: (agent: AgentHubData) => void,
  copyToClipboard: (text: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  publicPage: boolean = false,
): ColumnDef<AgentHubData>[] => [
  {
    header: "Agent Name",
    accessorKey: "name",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const agent = row.original;
      return (
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="font-medium text-sm">{agent.name}</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(agent.name)}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy agent name"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy agent name</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="md:hidden">
            <p className="text-xs text-muted-foreground">
              {agent.description}
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
    cell: ({ row }) => (
      <p className="text-xs line-clamp-2">{row.original.description || "-"}</p>
    ),
    meta: {
      className: "hidden md:table-cell",
    },
  },
  {
    header: "Version",
    accessorKey: "version",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <Badge className={cn("text-xs", badgeClass.blue)}>
        v{row.original.version}
      </Badge>
    ),
    meta: {
      className: "hidden lg:table-cell",
    },
  },
  {
    header: "Protocol",
    accessorKey: "protocolVersion",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <span className="text-xs">{row.original.protocolVersion || "-"}</span>
    ),
    meta: {
      className: "hidden lg:table-cell",
    },
  },
  {
    header: "Skills",
    accessorKey: "skills",
    enableSorting: false,
    cell: ({ row }) => {
      const agent = row.original;
      const skills = agent.skills || [];
      return (
        <div className="space-y-1">
          <span className="text-xs font-medium">
            {skills.length} skill{skills.length !== 1 ? "s" : ""}
          </span>
          {skills.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {skills.slice(0, 2).map((skill) => (
                <Badge
                  key={skill.id}
                  className={cn("text-xs", badgeClass.purple)}
                >
                  {skill.name}
                </Badge>
              ))}
              {skills.length > 2 && (
                <span className="text-xs text-muted-foreground">
                  +{skills.length - 2}
                </span>
              )}
            </div>
          )}
        </div>
      );
    },
  },
  {
    header: "Capabilities",
    accessorKey: "capabilities",
    enableSorting: false,
    cell: ({ row }) => {
      const agent = row.original;
      const capabilities = agent.capabilities || {};
      const capabilityList = Object.entries(capabilities)
        .filter(([, value]) => value === true)
        .map(([key]) => key);

      return (
        <div className="flex flex-wrap gap-1">
          {capabilityList.length === 0 ? (
            <span className="text-muted-foreground text-xs">-</span>
          ) : (
            capabilityList.map((capability) => (
              <Badge
                key={capability}
                className={cn("text-xs", badgeClass.emerald)}
              >
                {capability}
              </Badge>
            ))
          )}
        </div>
      );
    },
  },
  {
    header: "I/O Modes",
    accessorKey: "defaultInputModes",
    enableSorting: false,
    cell: ({ row }) => {
      const agent = row.original;
      const inputModes = agent.defaultInputModes || [];
      const outputModes = agent.defaultOutputModes || [];

      return (
        <div className="space-y-1">
          <div className="text-xs">
            <span className="font-medium">In:</span>{" "}
            {inputModes.join(", ") || "-"}
          </div>
          <div className="text-xs">
            <span className="font-medium">Out:</span>{" "}
            {outputModes.join(", ") || "-"}
          </div>
        </div>
      );
    },
    meta: {
      className: "hidden xl:table-cell",
    },
  },
  {
    header: "Public",
    accessorKey: "is_public",
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const publicA = rowA.original.is_public === true ? 1 : 0;
      const publicB = rowB.original.is_public === true ? 1 : 0;
      return publicA - publicB;
    },
    cell: ({ row }) =>
      row.original.is_public === true ? (
        <Badge className={cn("text-xs", badgeClass.emerald)}>Yes</Badge>
      ) : (
        <Badge className={cn("text-xs", badgeClass.gray)}>No</Badge>
      ),
    meta: {
      className: "hidden md:table-cell",
    },
  },
  {
    header: "Details",
    id: "details",
    enableSorting: false,
    cell: ({ row }) => {
      const agent = row.original;
      return (
        <Button
          size="sm"
          variant="secondary"
          onClick={() => showModal(agent)}
        >
          <Info className="h-3.5 w-3.5" />
          <span className="hidden lg:inline">Details</span>
          <span className="lg:hidden">Info</span>
        </Button>
      );
    },
  },
];
