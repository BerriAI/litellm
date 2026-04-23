import React, { useState } from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Badge } from "@/components/ui/badge";
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
  Trash2,
} from "lucide-react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  getGuardrailLogoAndName,
  guardrail_provider_map,
  skipSystemMessageToChoice,
} from "./guardrail_info_helpers";
import EditGuardrailForm from "./edit_guardrail_form";
import { Guardrail, GuardrailDefinitionLocation } from "./types";

interface GuardrailTableProps {
  guardrailsList: Guardrail[];
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
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  isAdmin = false,
  onGuardrailClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedGuardrail, setSelectedGuardrail] = useState<Guardrail | null>(
    null,
  );

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const handleEditSuccess = () => {
    setEditModalVisible(false);
    setSelectedGuardrail(null);
    onGuardrailUpdated();
  };

  const columns: ColumnDef<Guardrail>[] = [
    {
      header: "Guardrail ID",
      accessorKey: "guardrail_id",
      cell: (info) => {
        const v = String(info.getValue() || "");
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px] rounded"
                  onClick={() => v && onGuardrailClick(v)}
                >
                  {v ? `${v.slice(0, 7)}...` : ""}
                </button>
              </TooltipTrigger>
              <TooltipContent>{v}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Name",
      accessorKey: "guardrail_name",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs font-medium">
                  {guardrail.guardrail_name || "-"}
                </span>
              </TooltipTrigger>
              <TooltipContent>{guardrail.guardrail_name}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "litellm_params.guardrail",
      cell: ({ row }) => {
        const guardrail = row.original;
        const { logo, displayName } = getGuardrailLogoAndName(
          guardrail.litellm_params.guardrail,
        );
        return (
          <div className="flex items-center space-x-2">
            {logo && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={logo}
                alt={`${displayName} logo`}
                className="w-4 h-4"
                onError={(e) => {
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
            className={
              guardrail.litellm_params?.default_on
                ? "text-xs font-normal bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                : "text-xs font-normal bg-muted text-muted-foreground"
            }
          >
            {guardrail.litellm_params?.default_on
              ? "Default On"
              : "Default Off"}
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
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">
                  {formatDate(guardrail.created_at)}
                </span>
              </TooltipTrigger>
              <TooltipContent>{guardrail.created_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">
                  {formatDate(guardrail.updated_at)}
                </span>
              </TooltipTrigger>
              <TooltipContent>{guardrail.updated_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        const guardrail = row.original;
        const isConfigGuardrail =
          guardrail.guardrail_definition_location ===
          GuardrailDefinitionLocation.CONFIG;
        return (
          <div className="flex space-x-2 items-center">
            {isConfigGuardrail ? (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 cursor-not-allowed text-muted-foreground"
                      disabled
                      data-testid="config-delete-icon"
                      aria-label="Delete guardrail (config)"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Config guardrail cannot be deleted on the dashboard. Please
                    delete it from the config file.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive"
                      onClick={() =>
                        guardrail.guardrail_id &&
                        onDeleteClick(
                          guardrail.guardrail_id,
                          guardrail.guardrail_name || "Unnamed Guardrail",
                        )
                      }
                      aria-label="Delete guardrail"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Delete guardrail</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: guardrailsList,
    columns,
    state: { sorting },
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
                      header.id === "actions"
                        ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                        : ""
                    }`}
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
                      {header.id !== "actions" && (
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
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
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
                          ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
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
                    <p>No guardrails found</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {selectedGuardrail && (
        <EditGuardrailForm
          visible={editModalVisible}
          onClose={() => setEditModalVisible(false)}
          accessToken={accessToken}
          onSuccess={handleEditSuccess}
          guardrailId={selectedGuardrail.guardrail_id || ""}
          fullLitellmParams={selectedGuardrail.litellm_params}
          initialValues={{
            guardrail_name: selectedGuardrail.guardrail_name || "",
            provider:
              Object.keys(guardrail_provider_map).find(
                (key) =>
                  guardrail_provider_map[key] ===
                  selectedGuardrail?.litellm_params.guardrail,
              ) || "",
            mode: selectedGuardrail.litellm_params.mode,
            default_on: selectedGuardrail.litellm_params.default_on,
            pii_entities_config:
              selectedGuardrail.litellm_params.pii_entities_config,
            skip_system_message_choice: skipSystemMessageToChoice(
              selectedGuardrail.litellm_params?.skip_system_message_in_guardrail,
            ),
            ...selectedGuardrail.guardrail_info,
          }}
        />
      )}
    </div>
  );
};

export default GuardrailTable;
