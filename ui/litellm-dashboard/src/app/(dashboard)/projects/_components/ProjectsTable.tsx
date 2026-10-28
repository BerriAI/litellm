"use client";

import { SortingState } from "@tanstack/react-table";
import { FolderKanban } from "lucide-react";
import { useMemo, useState } from "react";

import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import { DataTable, DataTablePagination } from "@/components/shared/DataTable";

import { getProjectsTableColumns } from "./ProjectsTableColumns";
import { PROJECTS_DEFAULT_PAGE_SIZE, useProjectsTableState } from "./useProjectsUrlState";

interface ProjectsTableProps {
  projects: ProjectResponse[];
  isLoading: boolean;
  isFiltered: boolean;
  onProjectClick: (projectId: string) => void;
  teamAliasMap: Map<string, string>;
  isTeamsLoading: boolean;
}

const PAGE_SIZE_OPTIONS = [PROJECTS_DEFAULT_PAGE_SIZE, 25, 50];

function EmptyState({ isFiltered }: { isFiltered: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <FolderKanban className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {isFiltered ? "No matching projects" : "No projects yet"}
      </div>
      <div className="text-sm text-muted-foreground">
        {isFiltered ? "Try a different search term." : "Create a project to organize keys within your teams."}
      </div>
    </div>
  );
}

export function ProjectsTable({
  projects,
  isLoading,
  isFiltered,
  onProjectClick,
  teamAliasMap,
  isTeamsLoading,
}: ProjectsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const { pagination, onPaginationChange } = useProjectsTableState();
  const pageSize = PAGE_SIZE_OPTIONS.includes(pagination.pageSize) ? pagination.pageSize : PROJECTS_DEFAULT_PAGE_SIZE;

  const columns = useMemo(() => {
    const deps = { onProjectClick, teamAliasMap, isTeamsLoading };
    return getProjectsTableColumns(deps);
  }, [onProjectClick, teamAliasMap, isTeamsLoading]);

  const pageCount = Math.max(Math.ceil(projects.length / pageSize), 1);
  const pageIndex = pagination.pageIndex < pageCount ? pagination.pageIndex : 0;

  return (
    <DataTable
      data={projects}
      columns={columns}
      getRowId={(project, index) => project.project_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="client"
      pagination={{ pageIndex, pageSize }}
      pageSizeOptions={PAGE_SIZE_OPTIONS}
      paginationSlot={() => (
        <DataTablePagination
          page={pageIndex}
          pageSize={pageSize}
          rowCount={projects.length}
          onPageChange={(nextPageIndex) => onPaginationChange({ pageIndex: nextPageIndex, pageSize })}
          onPageSizeChange={(nextPageSize) => onPaginationChange({ pageIndex: 0, pageSize: nextPageSize })}
          pageSizeOptions={PAGE_SIZE_OPTIONS}
          isLoading={isLoading}
        />
      )}
      isLoading={isLoading}
      loadingMessage="Loading projects…"
      noDataMessage={<EmptyState isFiltered={isFiltered} />}
      size="compact"
    />
  );
}
