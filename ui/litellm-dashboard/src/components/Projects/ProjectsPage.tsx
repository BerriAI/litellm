import { useProjects, ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { PlusOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Flex,
  Input,
  Layout,
  Space,
  Table,
  Tag,
  theme,
  Tooltip,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { LayersIcon, SearchIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CreateProjectModal } from "./ProjectModals/CreateProjectModal";

const { Title, Text } = Typography;
const { Content } = Layout;

export function ProjectsPage() {
  const { token } = theme.useToken();
  const { data: projects, isLoading } = useProjects();
  const { data: teams } = useTeams();

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
        <Tooltip title={id}>
          <Text
            ellipsis
            className="text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs cursor-pointer"
            style={{ fontSize: 14, padding: "1px 8px" }}
          >
            {id}
          </Text>
        </Tooltip>
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
        const alias = teamAliasMap.get(record.team_id ?? "");
        return alias ?? record.team_id ?? "—";
      },
    },
    {
      title: "Models",
      key: "models",
      render: (_: unknown, record: ProjectResponse) => {
        const models = record.models ?? [];
        return (
          <Tooltip title={models.length > 0 ? models.join(", ") : "No models"}>
            <Tag color="blue" style={{ fontSize: 14, padding: "2px 8px", margin: 0 }}>
              <Flex align="center" gap={6}>
                <LayersIcon size={14} />
                {models.length}
              </Flex>
            </Tag>
          </Tooltip>
        );
      },
    },
    {
      title: "Status",
      dataIndex: "blocked",
      key: "status",
      render: (blocked: boolean) => (
        <Tag color={blocked ? "red" : "green"}>
          {blocked ? "Blocked" : "Active"}
        </Tag>
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

  return (
    <Content
      style={{ padding: token.paddingLG, paddingInline: token.paddingLG * 2 }}
    >
      <Flex
        justify="space-between"
        align="center"
        style={{ marginBottom: 16 }}
      >
        <Space direction="vertical" size={0}>
          <Title level={2} style={{ margin: 0 }}>
            Projects
          </Title>
          <Text type="secondary">
            Manage projects within your teams
          </Text>
        </Space>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setIsCreateModalVisible(true)}
        >
          Create Project
        </Button>
      </Flex>

      <Card styles={{ body: { padding: 0 } }}>
        <Flex
          justify="space-between"
          align="center"
          style={{ padding: "12px 16px" }}
        >
          <Input
            prefix={<SearchIcon size={16} />}
            placeholder="Search projects by name, ID, description, or team..."
            style={{ maxWidth: 400 }}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
          />
        </Flex>
        <Table
          columns={columns}
          dataSource={filteredProjects}
          rowKey="project_id"
          loading={isLoading}
          pagination={{
            current: currentPage,
            pageSize,
            total: filteredProjects.length,
            onChange: (page) => setCurrentPage(page),
            size: "small",
            showTotal: (total) => `${total} projects`,
            showSizeChanger: false,
          }}
        />
      </Card>

      <CreateProjectModal
        isOpen={isCreateModalVisible}
        onClose={() => setIsCreateModalVisible(false)}
      />
    </Content>
  );
}
