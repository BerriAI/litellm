import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { PlusOutlined } from "@ant-design/icons";
import { Button, Flex, Input, Layout, Space, theme, Typography } from "antd";
import { SearchIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { CreateProjectModal } from "./ProjectModals/CreateProjectModal";
import { ProjectDetail } from "./ProjectDetailsPage";
import { ProjectsTable } from "./ProjectsTable";

const { Title, Text } = Typography;
const { Content } = Layout;

export function ProjectsPage() {
  const { token } = theme.useToken();
  const { data: projects, isLoading } = useProjects();
  const { data: teams, isLoading: isTeamsLoading } = useTeams();

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
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
    return <ProjectDetail projectId={selectedProjectId} onBack={() => setSelectedProjectId(null)} />;
  }

  return (
    <Content style={{ padding: token.paddingLG, paddingInline: token.paddingLG * 2 }}>
      <Flex justify="space-between" align="center" style={{ marginBottom: 16 }}>
        <Space direction="vertical" size={0}>
          <Title level={2} style={{ margin: 0 }}>
            Projects
          </Title>
          <Text type="secondary">Manage projects within your teams</Text>
        </Space>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsCreateModalVisible(true)}>
          Create Project
        </Button>
      </Flex>

      <Flex align="center" style={{ marginBottom: 12 }}>
        <Input
          prefix={<SearchIcon size={16} />}
          placeholder="Search projects by name, ID, description, or team..."
          style={{ maxWidth: 400 }}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
        />
      </Flex>

      <ProjectsTable
        projects={filteredProjects}
        isLoading={isLoading}
        isFiltered={searchText.trim().length > 0}
        onProjectClick={setSelectedProjectId}
        teamAliasMap={teamAliasMap}
        isTeamsLoading={isTeamsLoading}
      />

      <CreateProjectModal isOpen={isCreateModalVisible} onClose={() => setIsCreateModalVisible(false)} />
    </Content>
  );
}
