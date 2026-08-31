import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Folder, Plus, SearchIcon, X } from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import { useMemo, useState } from "react";
import { PageHeader } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { CreateProjectModal } from "./ProjectModals/CreateProjectModal";
import { ProjectDetail } from "./ProjectDetailsPage";
import { ProjectsTable } from "./ProjectsTable";

export function ProjectsPage() {
  const { data: projects, isLoading } = useProjects();
  const { data: teams, isLoading: isTeamsLoading } = useTeams();

  const [selectedProjectId, setSelectedProjectId] = useQueryState(
    "project",
    parseAsString.withOptions({ history: "push" }),
  );
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");

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

  if (selectedProjectId) {
    return (
      <ProjectDetail
        projectId={selectedProjectId}
        onBack={() => void setSelectedProjectId(null, { history: "replace" })}
      />
    );
  }

  return (
    <div className="p-8">
      <PageHeader
        icon={<Folder />}
        title="Projects"
        subtitle="Manage projects within your teams"
        primaryAction={
          <Button onClick={() => setIsCreateModalVisible(true)}>
            <Plus className="size-4" />
            Create Project
          </Button>
        }
      />

      <div className="mt-6 mb-3 flex items-center">
        <InputGroup className="max-w-[400px]">
          <InputGroupAddon>
            <SearchIcon className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            placeholder="Search projects by name, ID, description, or team..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          {searchText && (
            <InputGroupAddon align="inline-end">
              <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => setSearchText("")}>
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          )}
        </InputGroup>
      </div>

      <ProjectsTable
        projects={filteredProjects}
        isLoading={isLoading}
        isFiltered={searchText.trim().length > 0}
        onProjectClick={(id) => void setSelectedProjectId(id)}
        teamAliasMap={teamAliasMap}
        isTeamsLoading={isTeamsLoading}
      />

      <CreateProjectModal isOpen={isCreateModalVisible} onClose={() => setIsCreateModalVisible(false)} />
    </div>
  );
}
