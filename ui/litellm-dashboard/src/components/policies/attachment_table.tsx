import React, { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow, Icon, Badge } from "@tremor/react";
import { TrashIcon, SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { Tooltip, Tag } from "antd";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { PolicyAttachment } from "./types";
import ImpactPopover from "./impact_popover";

interface AttachmentTableProps {
  attachments: PolicyAttachment[];
  isLoading: boolean;
  onDeleteClick: (attachmentId: string) => void;
  isAdmin: boolean;
  accessToken: string | null;
}

const AttachmentTable: React.FC<AttachmentTableProps> = ({
  attachments,
  isLoading,
  onDeleteClick,
  isAdmin,
  accessToken,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const columns: ColumnDef<PolicyAttachment>[] = [
    {
      header: "Attachment ID",
      accessorKey: "attachment_id",
      cell: (info: any) => (
        <Tooltip title={String(info.getValue() || "")}>
          <span className="font-mono text-xs text-gray-600">
            {info.getValue() ? `${String(info.getValue()).slice(0, 7)}...` : ""}
          </span>
        </Tooltip>
      ),
    },
    {
      header: "Policy",
      accessorKey: "policy_name",
      cell: ({ row }) => {
        const attachment = row.original;
        return (
          <Badge color="blue" size="xs">
            {attachment.policy_name}
          </Badge>
        );
      },
    },
    {
      header: "Scope",
      accessorKey: "scope",
      cell: ({ row }) => {
        const attachment = row.original;
        if (attachment.scope === "*") {
          return (
            <Badge color="amber" size="xs">
              Global (*)
            </Badge>
          );
        }
        return attachment.scope ? (
          <span className="text-xs">{attachment.scope}</span>
        ) : (
          <span className="text-xs text-gray-400">-</span>
        );
      },
    },
    {
      header: "Teams",
      accessorKey: "teams",
      cell: ({ row }) => {
        const attachment = row.original;
        const teams = attachment.teams || [];
        if (teams.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {teams.slice(0, 2).map((t, i) => (
              <Tag key={i} color="cyan" className="text-xs">
                {t}
              </Tag>
            ))}
            {teams.length > 2 && (
              <Tooltip title={teams.slice(2).join(", ")}>
                <Tag className="text-xs">+{teams.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Keys",
      accessorKey: "keys",
      cell: ({ row }) => {
        const attachment = row.original;
        const keys = attachment.keys || [];
        if (keys.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {keys.slice(0, 2).map((k, i) => (
              <Tag key={i} color="purple" className="text-xs">
                {k}
              </Tag>
            ))}
            {keys.length > 2 && (
              <Tooltip title={keys.slice(2).join(", ")}>
                <Tag className="text-xs">+{keys.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Models",
      accessorKey: "models",
      cell: ({ row }) => {
        const attachment = row.original;
        const models = attachment.models || [];
        if (models.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {models.slice(0, 2).map((m, i) => (
              <Tag key={i} color="green" className="text-xs">
                {m}
              </Tag>
            ))}
            {models.length > 2 && (
              <Tooltip title={models.slice(2).join(", ")}>
                <Tag className="text-xs">+{models.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Tags",
      accessorKey: "tags",
      cell: ({ row }) => {
        const attachment = row.original;
        const tags = attachment.tags || [];
        if (tags.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {tags.slice(0, 2).map((t, i) => (
              <Tag key={i} color="orange" className="text-xs">
                {t}
              </Tag>
            ))}
            {tags.length > 2 && (
              <Tooltip title={tags.slice(2).join(", ")}>
                <Tag className="text-xs">+{tags.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const attachment = row.original;
        return (
          <Tooltip title={attachment.created_at}>
            <span className="text-xs">{formatDate(attachment.created_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        const attachment = row.original;
        return (
          <div className="flex space-x-2">
            <ImpactPopover attachment={attachment} accessToken={accessToken} />
            {isAdmin && (
              <Tooltip title="Delete attachment">
                <Icon
                  icon={TrashIcon}
                  size="sm"
                  onClick={() => onDeleteClick(attachment.attachment_id)}
                  className="cursor-pointer hover:text-red-500"
                />
              </Tooltip>
            )}
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: attachments,
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
                    className={`py-1 h-8 ${
                      header.id === "actions" ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]" : ""
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                      {header.id !== "actions" && (
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
                      )}
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
            ) : attachments.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === "actions"
                          ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>No attachments found</p>
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

export default AttachmentTable;
