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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Table } from "antd";
import type { ColumnsType } from "antd/es/table";
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

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
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

  // ---------- filtered data ----------
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

  // ---------- Ant Design columns ----------
  const columns: ColumnsType<ProjectResponse> = [
    {
      title: "ID",
      dataIndex: "project_id",
      key: "project_id",
      width: 170,
      render: (id: string) => (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                onClick={() => setSelectedProjectId(id)}
                className="text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-sm cursor-pointer px-2 py-0.5 rounded inline-block truncate max-w-[160px]"
              >
                {id}
              </span>
            </TooltipTrigger>
            <TooltipContent>{id}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ),
    },
    {
      title: "Name",
      dataIndex: "project_alias",
      key: "project_alias",
      sorter: (a, b) => (a.project_alias ?? "").localeCompare(b.project_alias ?? ""),
      render: (alias: string | null) => alias ?? "—",
    },
    {
      title: "Team",
      key: "team",
      sorter: (a, b) => {
        const aAlias = teamAliasMap.get(a.team_id ?? "") ?? "";
        const bAlias = teamAliasMap.get(b.team_id ?? "") ?? "";
        return aAlias.localeCompare(bAlias);
      },
      render: (_: unknown, record: ProjectResponse) => {
        if (!record.team_id) return "—";
        const alias = teamAliasMap.get(record.team_id);
        if (alias) return alias;
        if (isTeamsLoading)
          return <Loader2 className="h-3.5 w-3.5 animate-spin" />;
        return record.team_id;
      },
    },
    {
      title: "Models",
      key: "models",
      render: (_: unknown, record: ProjectResponse) => {
        const models = record.models ?? [];
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
      title: "Status",
      dataIndex: "blocked",
      key: "status",
      render: (blocked: boolean) => (
        <Badge
          className={
            blocked
              ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
              : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
          }
        >
          {blocked ? "Blocked" : "Active"}
        </Badge>
      ),
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      sorter: (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      responsive: ["lg"],
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: "Updated",
      dataIndex: "updated_at",
      key: "updated_at",
      responsive: ["xl"],
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
  ];

  if (selectedProjectId) {
    return (
      <ProjectDetail
        projectId={selectedProjectId}
        onBack={() => setSelectedProjectId(null)}
      />
    );
  }

  const totalPages = Math.max(1, Math.ceil(filteredProjects.length / pageSize));

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
              {filteredProjects.length} projects
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
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <Table
          columns={columns}
          dataSource={filteredProjects.slice(
            (currentPage - 1) * pageSize,
            currentPage * pageSize,
          )}
          rowKey="project_id"
          loading={isLoading}
          pagination={false}
        />
      </Card>

      <CreateProjectModal
        isOpen={isCreateModalVisible}
        onClose={() => setIsCreateModalVisible(false)}
      />
    </main>
  );
}
