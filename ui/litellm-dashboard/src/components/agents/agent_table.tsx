import React, { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Copy,
  Trash2,
} from "lucide-react";
import { Agent } from "./types";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";

interface AgentTableProps {
  agentsList: Agent[];
  isLoading: boolean;
  onDeleteClick: (agentId: string, agentName: string) => void;
  accessToken: string | null;
  onAgentUpdated: () => void;
  isAdmin: boolean;
  onAgentClick: (agentId: string) => void;
}

const AgentTable: React.FC<AgentTableProps> = ({
  agentsList,
  isLoading,
  onDeleteClick,
  isAdmin,
  onAgentClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const columns: ColumnDef<Agent>[] = [
    {
      header: "Agent Name",
      accessorKey: "agent_name",
      cell: ({ row }) => {
        const agent = row.original;
        const name = agent.agent_name || "";
        return (
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => onAgentClick(agent.agent_id)}
                    className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate min-w-[200px] rounded"
                  >
                    {name}
                  </button>
                </TooltipTrigger>
                <TooltipContent>{name}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboard(agent.agent_id);
                    }}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy Agent ID"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy Agent ID</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "agent_card_params.description",
      cell: ({ row }) => {
        const description =
          row.original.agent_card_params?.description || "No description";
        return (
          <span className="text-xs text-muted-foreground block max-w-[300px] truncate">
            {description}
          </span>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const agent = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{formatDate(agent.created_at)}</span>
              </TooltipTrigger>
              <TooltipContent>{agent.created_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            enableSorting: false,
            cell: ({ row }: { row: { original: Agent } }) => {
              const agent = row.original;

              return (
                <div className="flex items-center gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteClick(agent.agent_id, agent.agent_name);
                          }}
                          aria-label="Delete agent"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Delete agent</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              );
            },
          },
        ]
      : []),
  ];

  const table = useReactTable({
    data: agentsList,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className="py-1 h-8"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                      </div>
                      <div className="w-4">
                        {header.column.getIsSorted() ? (
                          {
                            asc: (
                              <ChevronUp className="h-4 w-4 text-primary" />
                            ),
                            desc: (
                              <ChevronDown className="h-4 w-4 text-primary" />
                            ),
                          }[header.column.getIsSorted() as string]
                        ) : (
                          <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : agentsList && agentsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>No agents found. Create one to get started.</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default AgentTable;
