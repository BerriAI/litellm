import React, { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow, Button } from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon, TrashIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { CopyOutlined } from "@ant-design/icons";
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
  accessToken,
  onAgentUpdated,
  isAdmin,
  onAgentClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

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
            <Tooltip title={name}>
                <Button
                  size="xs"
                  variant="light"
                className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate min-w-[200px] justify-start"
                  onClick={() => onAgentClick(agent.agent_id)}
                >
                {name}
                </Button>
              </Tooltip>
            <Tooltip title="Copy Agent ID">
              <CopyOutlined
                onClick={(e) => {
                  e.stopPropagation();
                  copyToClipboard(agent.agent_id);
                }}
                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
              />
            </Tooltip>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "agent_card_params.description",
      cell: ({ row }) => {
        const description = row.original.agent_card_params?.description || "No description";
        return (
          <span className="text-xs text-gray-600 block max-w-[300px] truncate">
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
          <Tooltip title={agent.created_at}>
            <span className="text-xs">{formatDate(agent.created_at)}</span>
          </Tooltip>
        );
      },
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            enableSorting: false,
            cell: ({ row }: any) => {
              const agent = row.original;
              
              return (
                <div className="flex items-center gap-1">
                  <Tooltip title="Delete agent">
                    <Button
                      size="xs"
                      variant="light"
                      color="red"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteClick(agent.agent_id, agent.agent_name);
                      }}
                      icon={TrashIcon}
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                    />
                  </Tooltip>
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
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell
                    key={header.id}
                    className="py-1 h-8"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                      <div className="w-4">
                        {header.column.getIsSorted() ? (
                          {
                            asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                            desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />,
                          }[header.column.getIsSorted() as string]
                        ) : (
                          <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                        )}
                      </div>
                    </div>
                  </TableHeaderCell>
                ))}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : agentsList && agentsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
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
