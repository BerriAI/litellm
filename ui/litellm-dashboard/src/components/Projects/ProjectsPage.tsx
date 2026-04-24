import {
  useProjects,
  ProjectResponse,
} from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
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
import {
  SortState,
  TableHeaderSortDropdown,
} from "../common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { cn } from "@/lib/utils";
import {
  ChevronLeft,
  ChevronRight,
  Layers,
  Loader2,
  Plus,
  Search,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CreateProjectModal } from "./ProjectModals/CreateProjectModal";
import { ProjectDetail } from "./ProjectDetailsPage";

export function ProjectsPage() {
  const { data: projects, isLoading } = useProjects();
  const { data: teams, isLoading: isTeamsLoading } = useTeams();

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    null,
  );
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [sorting, setSorting] = useState<SortingState>([]);
  const pageSize = 10;

  useEffect(() => {
    setCurrentPage(1);
  }, [searchText]);

  // Build a team_id → team_alias lookup from the teams list
  const teamAliasMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const team of teams ?? []) {
      map.set(team.team_id, team.team_alias ?? team.team_id);
    }
    return map;
  }, [teams]);

  const filteredProjects = useMemo(() => {
    const list = projects ?? [];
    if (!searchText) return list;
    const lower = searchText.toLowerCase();
    return list.filter((p) => {
      const alias = teamAliasMap.get(p.team_id ?? "") ?? "";
      return (
        (p.project_alias ?? "").toLowerCase().includes(lower) ||
        p.project_id.toLowerCase().includes(lower) ||
        (p.description ?? "").toLowerCase().includes(lower) ||
        alias.toLowerCase().includes(lower)
      );
    });
  }, [projects, searchText, teamAliasMap]);

  const columnDefs = useMemo<ColumnDef<ProjectResponse>[]>(
    () => [
      {
        id: "project_id",
        accessorKey: "project_id",
        header: () => <span>ID</span>,
        enableSorting: false,
        size: 170,
        cell: ({ row }) => {
          const id = row.original.project_id;
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => setSelectedProjectId(id)}
                    className="text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-sm cursor-pointer px-2 py-0.5 rounded inline-block truncate max-w-[160px]"
                  >
                    {id}
                  </button>
                </TooltipTrigger>
                <TooltipContent>{id}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        },
      },
      {
        id: "project_alias",
        accessorKey: "project_alias",
        header: () => <span>Name</span>,
        enableSorting: true,
        sortingFn: (a, b) =>
          (a.original.project_alias ?? "").localeCompare(
            b.original.project_alias ?? "",
          ),
        cell: ({ getValue }) => (getValue() as string | null) ?? "—",
      },
      {
        id: "team",
        header: () => <span>Team</span>,
        enableSorting: true,
        sortingFn: (a, b) => {
          const aAlias = teamAliasMap.get(a.original.team_id ?? "") ?? "";
          const bAlias = teamAliasMap.get(b.original.team_id ?? "") ?? "";
          return aAlias.localeCompare(bAlias);
        },
        cell: ({ row }) => {
          const record = row.original;
          if (!record.team_id) return "—";
          const alias = teamAliasMap.get(record.team_id);
          if (alias) return alias;
          if (isTeamsLoading)
            return <Loader2 className="h-3.5 w-3.5 animate-spin" />;
          return record.team_id;
        },
      },
      {
        id: "models",
        header: () => <span>Models</span>,
        enableSorting: false,
        cell: ({ row }) => {
          const models = row.original.models ?? [];
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 gap-1.5">
                    <Layers className="h-3.5 w-3.5" />
                    {models.length}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  {models.length > 0 ? models.join(", ") : "No models"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        },
      },
      {
        id: "status",
        accessorKey: "blocked",
        header: () => <span>Status</span>,
        enableSorting: false,
        cell: ({ row }) => (
          <Badge
            className={
              row.original.blocked
                ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
            }
          >
            {row.original.blocked ? "Blocked" : "Active"}
          </Badge>
        ),
      },
      {
        id: "created_at",
        accessorKey: "created_at",
        header: () => <span>Created</span>,
        enableSorting: true,
        sortingFn: (a, b) =>
          new Date(a.original.created_at).getTime() -
          new Date(b.original.created_at).getTime(),
        cell: ({ getValue }) =>
          new Date(getValue() as string).toLocaleDateString(),
      },
      {
        id: "updated_at",
        accessorKey: "updated_at",
        header: () => <span>Updated</span>,
        enableSorting: false,
        cell: ({ getValue }) =>
          new Date(getValue() as string).toLocaleDateString(),
      },
    ],
    [teamAliasMap, isTeamsLoading],
  );

  const table = useReactTable<ProjectResponse>({
    data: filteredProjects,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: (row) => row.project_id,
  });

  const sortedRows = table.getRowModel().rows;
  const paginatedRows = sortedRows.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize,
  );
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));

  if (selectedProjectId) {
    return (
      <ProjectDetail
        projectId={selectedProjectId}
        onBack={() => setSelectedProjectId(null)}
      />
    );
  }

  return (
    <main className="p-6 md:px-12">
      <div className="flex justify-between items-center mb-4">
        <div className="flex flex-col">
          <h2 className="text-2xl font-semibold m-0">Projects</h2>
          <p className="text-muted-foreground">
            Manage projects within your teams
          </p>
        </div>
        <Button onClick={() => setIsCreateModalVisible(true)}>
          <Plus className="h-4 w-4" />
          Create Project
        </Button>
      </div>

      <Card className="p-0">
        <div className="flex justify-between items-center px-4 py-3">
          <div className="relative max-w-[400px] w-full">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Search projects by name, ID, description, or team..."
              className="pl-8"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-1 text-sm">
            <span className="text-muted-foreground">
              {sortedRows.length} projects
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-muted-foreground">
              {currentPage}/{totalPages}
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              disabled={currentPage >= totalPages}
              onClick={() =>
                setCurrentPage((p) => Math.min(totalPages, p + 1))
              }
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
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
                              isSorted === false
                                ? false
                                : (isSorted as SortState)
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
                  No projects found
                </TableCell>
              </TableRow>
            ) : (
              paginatedRows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      <CreateProjectModal
        isOpen={isCreateModalVisible}
        onClose={() => setIsCreateModalVisible(false)}
      />
    </main>
  );
}
