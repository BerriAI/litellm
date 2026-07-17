"use client";

import { SortingState } from "@tanstack/react-table";
import { FolderKanban } from "lucide-react";
import { useMemo, useState } from "react";

import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import { DataTable } from "@/components/shared/DataTable";

import { getProjectsTableColumns } from "./ProjectsTableColumns";

interface ProjectsTableProps {
  projects: ProjectResponse[];
  isLoading: boolean;
  isFiltered: boolean;
  onProjectClick: (projectId: string) => void;
  teamAliasMap: Map<string, string>;
  isTeamsLoading: boolean;
}

const PAGE_SIZE_OPTIONS = [10, 25, 50];

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

  const columns = useMemo(() => {
    const deps = { onProjectClick, teamAliasMap, isTeamsLoading };
    return getProjectsTableColumns(deps);
  }, [onProjectClick, teamAliasMap, isTeamsLoading]);

  return (
    <DataTable
      data={projects}
      columns={columns}
      getRowId={(project, index) => project.project_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="client"
      pageSizeOptions={PAGE_SIZE_OPTIONS}
      isLoading={isLoading}
      loadingMessage="Loading projects…"
      noDataMessage={<EmptyState isFiltered={isFiltered} />}
      size="compact"
    />
  );
}
