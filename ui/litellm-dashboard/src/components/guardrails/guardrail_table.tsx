import React, { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow, Icon, Button } from "@tremor/react";
import { TrashIcon, SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { Badge } from "@tremor/react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { getGuardrailLogoAndName, guardrail_provider_map } from "./guardrail_info_helpers";
import EditGuardrailForm from "./edit_guardrail_form";

interface GuardrailItem {
  guardrail_id?: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
    pii_entities_config?: { [key: string]: string };
    [key: string]: any;
  };
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}

interface GuardrailTableProps {
  guardrailsList: GuardrailItem[];
  isLoading: boolean;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
  accessToken: string | null;
  onGuardrailUpdated: () => void;
  isAdmin?: boolean;
  onGuardrailClick: (id: string) => void;
}

const GuardrailTable: React.FC<GuardrailTableProps> = ({
  guardrailsList,
  isLoading,
  onDeleteClick,
  accessToken,
  onGuardrailUpdated,
  isAdmin = false,
  onGuardrailClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedGuardrail, setSelectedGuardrail] = useState<GuardrailItem | null>(null);

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const handleEditClick = (guardrail: GuardrailItem) => {
    setSelectedGuardrail(guardrail);
    setEditModalVisible(true);
  };

  const handleEditSuccess = () => {
    setEditModalVisible(false);
    setSelectedGuardrail(null);
    onGuardrailUpdated();
  };

  const columns: ColumnDef<GuardrailItem>[] = [
    {
      header: "Guardrail ID",
      accessorKey: "guardrail_id",
      cell: (info: any) => (
        <Tooltip title={String(info.getValue() || "")}>
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => info.getValue() && onGuardrailClick(info.getValue())}
          >
            {info.getValue() ? `${String(info.getValue()).slice(0, 7)}...` : ""}
          </Button>
        </Tooltip>
      ),
    },
    {
      header: "Name",
      accessorKey: "guardrail_name",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <Tooltip title={guardrail.guardrail_name}>
            <span className="text-xs font-medium">{guardrail.guardrail_name || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "litellm_params.guardrail",
      cell: ({ row }) => {
        const guardrail = row.original;
        const { logo, displayName } = getGuardrailLogoAndName(guardrail.litellm_params.guardrail);
        return (
          <div className="flex items-center space-x-2">
            {logo && (
              <img
                src={logo}
                alt={`${displayName} logo`}
                className="w-4 h-4"
                onError={(e) => {
                  // Hide broken image
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            )}
            <span className="text-xs">{displayName}</span>
          </div>
        );
      },
    },
    {
      header: "Mode",
      accessorKey: "litellm_params.mode",
      cell: ({ row }) => {
        const guardrail = row.original;
        return <span className="text-xs">{guardrail.litellm_params.mode}</span>;
      },
    },
    {
      header: "Default On",
      accessorKey: "litellm_params.default_on",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <Badge
            color={guardrail.litellm_params?.default_on ? "green" : "gray"}
            className="text-xs font-normal"
            size="xs"
          >
            {guardrail.litellm_params?.default_on ? "Default On" : "Default Off"}
          </Badge>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <Tooltip title={guardrail.created_at}>
            <span className="text-xs">{formatDate(guardrail.created_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <Tooltip title={guardrail.updated_at}>
            <span className="text-xs">{formatDate(guardrail.updated_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <div className="flex space-x-2">
            <Icon
              icon={TrashIcon}
              size="sm"
              onClick={() =>
                guardrail.guardrail_id &&
                onDeleteClick(guardrail.guardrail_id, guardrail.guardrail_name || "Unnamed Guardrail")
              }
              className="cursor-pointer hover:text-red-500"
              tooltip="Delete guardrail"
            />
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: guardrailsList,
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
            ) : guardrailsList.length > 0 ? (
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
                    <p>No guardrails found</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Edit Modal */}
      {selectedGuardrail && (
        <EditGuardrailForm
          visible={editModalVisible}
          onClose={() => setEditModalVisible(false)}
          accessToken={accessToken}
          onSuccess={handleEditSuccess}
          guardrailId={selectedGuardrail.guardrail_id || ""}
          initialValues={{
            guardrail_name: selectedGuardrail.guardrail_name || "",
            provider:
              Object.keys(guardrail_provider_map).find(
                (key) => guardrail_provider_map[key] === selectedGuardrail?.litellm_params.guardrail,
              ) || "",
            mode: selectedGuardrail.litellm_params.mode,
            default_on: selectedGuardrail.litellm_params.default_on,
            pii_entities_config: selectedGuardrail.litellm_params.pii_entities_config,
            ...selectedGuardrail.guardrail_info,
          }}
        />
      )}
    </div>
  );
};

export default GuardrailTable;
