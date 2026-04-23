import {
  AccessGroupResponse,
  useAccessGroups,
} from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useDeleteAccessGroup } from "@/app/(dashboard)/hooks/accessGroups/useDeleteAccessGroup";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { BotIcon, LayersIcon, Plus, SearchIcon, ServerIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import {
  SortState,
  TableHeaderSortDropdown,
} from "../common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import { AccessGroupDetail } from "./AccessGroupsDetailsPage";
import { AccessGroupCreateModal } from "./AccessGroupsModal/AccessGroupCreateModal";
import { AccessGroup } from "./types";

function mapResponseToAccessGroup(r: AccessGroupResponse): AccessGroup {
  return {
    id: r.access_group_id,
    name: r.access_group_name,
    description: r.description ?? "",
    modelIds: r.access_model_names,
    mcpServerIds: r.access_mcp_server_ids,
    agentIds: r.access_agent_ids,
    keyIds: r.assigned_key_ids,
    teamIds: r.assigned_team_ids,
    createdAt: r.created_at,
    createdBy: r.created_by ?? "",
    updatedAt: r.updated_at,
    updatedBy: r.updated_by ?? "",
  };
}

export function AccessGroupsPage() {
  const { data: groupsData, isLoading } = useAccessGroups();
  const groups = useMemo(
    () => (groupsData ?? []).map(mapResponseToAccessGroup),
    [groupsData],
  );

  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [groupToDelete, setGroupToDelete] = useState<AccessGroup | null>(null);
  const deleteMutation = useDeleteAccessGroup();
  const pageSize = 10;

  useEffect(() => {
    setCurrentPage(1);
  }, [searchText]);

  const filteredGroups = useMemo(
    () =>
      groups.filter(
        (group) =>
          group.name.toLowerCase().includes(searchText.toLowerCase()) ||
          group.id.toLowerCase().includes(searchText.toLowerCase()) ||
          group.description.toLowerCase().includes(searchText.toLowerCase()),
      ),
    [groups, searchText],
  );

  const columnDefs = useMemo<ColumnDef<AccessGroup>[]>(
    () => [
      {
        id: "id",
        accessorKey: "id",
        header: () => <span>ID</span>,
        enableSorting: false,
        size: 170,
        cell: ({ row }) => {
          const record = row.original;
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="truncate text-primary bg-primary/10 hover:bg-primary/20 text-xs cursor-pointer px-2 py-0.5 rounded max-w-full text-left"
                    style={{ fontSize: 14 }}
                    onClick={() => setSelectedGroupId(record.id)}
                  >
                    {record.id}
                  </button>
                </TooltipTrigger>
                <TooltipContent>{record.id}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        },
      },
      {
        id: "name",
        accessorKey: "name",
        header: () => <span>Name</span>,
        enableSorting: true,
        cell: ({ getValue }) => getValue() as string,
      },
      {
        id: "resources",
        header: () => <span>Resources</span>,
        enableSorting: false,
        cell: ({ row }) => {
          const record = row.original;
          const modelIds = record.modelIds ?? [];
          const mcpServerIds = record.mcpServerIds ?? [];
          const agentIds = record.agentIds ?? [];
          return (
            <TooltipProvider>
              <div className="flex items-center gap-3">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="flex items-center gap-1.5 px-2 py-0.5 text-sm">
                      <LayersIcon size={14} />
                      {modelIds?.length}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>{`${modelIds?.length} Models`}</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="flex items-center gap-1.5 px-2 py-0.5 text-sm">
                      <ServerIcon size={14} />
                      {mcpServerIds?.length}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>{`${mcpServerIds?.length} MCP Servers`}</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="flex items-center gap-1.5 px-2 py-0.5 text-sm">
                      <BotIcon size={14} />
                      {agentIds?.length}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>{`${agentIds?.length} Agents`}</TooltipContent>
                </Tooltip>
              </div>
            </TooltipProvider>
          );
        },
      },
      {
        id: "createdAt",
        accessorKey: "createdAt",
        header: () => <span>Created</span>,
        enableSorting: true,
        sortingFn: "datetime",
        cell: ({ getValue }) =>
          new Date(getValue() as string).toLocaleDateString(),
      },
      {
        id: "updatedAt",
        accessorKey: "updatedAt",
        header: () => <span>Updated</span>,
        enableSorting: false,
        cell: ({ getValue }) =>
          new Date(getValue() as string).toLocaleDateString(),
      },
      {
        id: "actions",
        header: () => <span>Actions</span>,
        enableSorting: false,
        cell: ({ row }) => (
          <div className="flex gap-2">
            <TableIconActionButton
              variant="Delete"
              tooltipText="Delete access group"
              onClick={() => setGroupToDelete(row.original)}
            />
          </div>
        ),
      },
    ],
    [],
  );

  const table = useReactTable<AccessGroup>({
    data: filteredGroups,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: (row) => row.id,
  });

  const sortedRows = table.getRowModel().rows;
  const paginatedRows = sortedRows.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize,
  );
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));

  if (selectedGroupId) {
    return (
      <AccessGroupDetail
        accessGroupId={selectedGroupId}
        onBack={() => setSelectedGroupId(null)}
      />
    );
  }

  return (
    <div className="p-6 md:px-12">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-2xl font-semibold m-0">Access Groups</h2>
          <p className="text-muted-foreground text-sm m-0">
            Manage resource permissions for your organization
          </p>
        </div>
        <Button onClick={() => setIsCreateModalVisible(true)}>
          <Plus className="h-4 w-4" />
          Create Access Group
        </Button>
      </div>

      <Card>
        <div className="flex justify-between items-center px-4 py-3 gap-3">
          <div className="relative max-w-md w-full">
            <SearchIcon
              size={16}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none"
            />
            <Input
              placeholder="Search groups by name, ID, or description..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              className="pl-9"
            />
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>{sortedRows.length} groups</span>
            {totalPages > 1 && (
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  Prev
                </Button>
                <span className="tabular-nums">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage >= totalPages}
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        </div>
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const canSort = header.column.getCanSort();
                  const isSorted = header.column.getIsSorted();
                  return (
                    <TableHead
                      key={header.id}
                      style={{ width: header.column.columnDef.size }}
                    >
                      <div className={cn("flex items-center gap-1")}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                        {canSort && (
                          <TableHeaderSortDropdown
                            sortState={
                              isSorted === false ? false : (isSorted as SortState)
                            }
                            onSortChange={(newState) => {
                              if (newState === false) {
                                setSorting([]);
                              } else {
                                setSorting([
                                  {
                                    id: header.column.id,
                                    desc: newState === "desc",
                                  },
                                ]);
                              }
                            }}
                            columnId={header.column.id}
                          />
                        )}
                      </div>
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={columnDefs.length}
                  className="text-center text-muted-foreground py-8"
                >
                  Loading…
                </TableCell>
              </TableRow>
            ) : paginatedRows.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={columnDefs.length}
                  className="text-center text-muted-foreground py-8"
                >
                  No access groups found
                </TableCell>
              </TableRow>
            ) : (
              paginatedRows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      <AccessGroupCreateModal
        visible={isCreateModalVisible}
        onCancel={() => setIsCreateModalVisible(false)}
      />

      <DeleteResourceModal
        isOpen={!!groupToDelete}
        title="Delete Access Group"
        message="Are you sure you want to delete this access group? This action cannot be undone."
        resourceInformationTitle="Access Group Information"
        resourceInformation={[
          { label: "ID", value: groupToDelete?.id, code: true },
          { label: "Name", value: groupToDelete?.name },
          { label: "Description", value: groupToDelete?.description || "—" },
        ]}
        onCancel={() => setGroupToDelete(null)}
        onOk={() => {
          if (!groupToDelete) return;
          deleteMutation.mutate(groupToDelete.id, {
            onSuccess: () => {
              setGroupToDelete(null);
            },
          });
        }}
        confirmLoading={deleteMutation.isPending}
      />
    </div>
  );
}
