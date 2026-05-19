/**
 * XCT Skills — list + detail (S2-08).
 *
 * antd only (per CLAUDE.md "no new @tremor/react imports"). Mounts as a tab
 * inside whatever skills shell wires it up; takes accessToken + isAdmin.
 */

import React, { useEffect, useState } from "react";
import {
  Button,
  Card,
  Drawer,
  Input,
  Modal,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from "antd";
import {
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  deleteXCTSkill,
  getXCTSkill,
  listXCTSkills,
  publishXCTSkill,
} from "../networking";
import SkillUploadForm from "./skill_upload_form";

const { Text, Title, Paragraph } = Typography;

interface SkillRow {
  skill_id: string;
  display_title?: string;
  description?: string;
  version?: string;
  source?: string;
  team_id?: string;
  user_id?: string;
  is_public?: boolean;
  xct_metadata?: Record<string, any>;
  updated_at?: string;
  created_by?: string;
  tool_schema?: any[];
  system_prompt_template?: string;
}

interface Props {
  accessToken: string;
  isAdmin?: boolean;
}

const SkillTable: React.FC<Props> = ({ accessToken, isAdmin = false }) => {
  const [rows, setRows] = useState<SkillRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [drawerSkill, setDrawerSkill] = useState<SkillRow | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);

  const fetchSkills = async (q?: string) => {
    setLoading(true);
    try {
      const data = await listXCTSkills(accessToken, q ? { q } : {});
      setRows(data?.data ?? []);
    } catch (e: any) {
      message.error(`Failed to load skills: ${e.message ?? e}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (accessToken) fetchSkills();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const handleView = async (skillId: string) => {
    try {
      const full = await getXCTSkill(accessToken, skillId);
      setDrawerSkill(full);
    } catch (e: any) {
      message.error(`Failed to load skill: ${e.message ?? e}`);
    }
  };

  const handleDelete = (skillId: string, title?: string) => {
    Modal.confirm({
      title: `Delete skill "${title ?? skillId}"?`,
      content: "This cannot be undone. Outstanding chat completions using this skill will fail.",
      okText: "Delete",
      okType: "danger",
      onOk: async () => {
        try {
          await deleteXCTSkill(accessToken, skillId);
          message.success("Skill deleted");
          fetchSkills(search);
        } catch (e: any) {
          message.error(`Delete failed: ${e.message ?? e}`);
        }
      },
    });
  };

  const handlePublish = async (skillId: string) => {
    try {
      await publishXCTSkill(accessToken, skillId);
      message.success("Skill published — content fields are now immutable");
      fetchSkills(search);
    } catch (e: any) {
      message.error(`Publish failed: ${e.message ?? e}`);
    }
  };

  const columns = [
    {
      title: "Title",
      dataIndex: "display_title",
      key: "display_title",
      render: (v: string | undefined, row: SkillRow) => (
        <Space direction="vertical" size={0}>
          <Text strong>{v ?? "(untitled)"}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {row.skill_id}
          </Text>
        </Space>
      ),
    },
    {
      title: "Version",
      dataIndex: "version",
      key: "version",
      width: 100,
      render: (v?: string, row?: SkillRow) => (
        <Space>
          <Tag>{v ?? "1"}</Tag>
          {row?.xct_metadata?.published ? <Tag color="gold">published</Tag> : null}
        </Space>
      ),
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      width: 110,
      render: (v?: string) => <Tag color={v === "custom" ? "blue" : "default"}>{v ?? "custom"}</Tag>,
    },
    {
      title: "Public",
      dataIndex: "is_public",
      key: "is_public",
      width: 90,
      render: (v?: boolean) => (v ? <Tag color="green">public</Tag> : <Tag>private</Tag>),
    },
    {
      title: "Team",
      dataIndex: "team_id",
      key: "team_id",
      ellipsis: true,
    },
    {
      title: "Owner",
      dataIndex: "created_by",
      key: "created_by",
      ellipsis: true,
    },
    {
      title: "Actions",
      key: "actions",
      width: 180,
      render: (_: any, row: SkillRow) => (
        <Space>
          <Tooltip title="View">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handleView(row.skill_id)}
            />
          </Tooltip>
          {!row.xct_metadata?.published && isAdmin ? (
            <Button size="small" onClick={() => handlePublish(row.skill_id)}>
              Publish
            </Button>
          ) : null}
          {isAdmin ? (
            <Tooltip title="Delete">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                onClick={() => handleDelete(row.skill_id, row.display_title)}
              />
            </Tooltip>
          ) : null}
        </Space>
      ),
    },
  ];

  return (
    <Card
      title={<Title level={4}>XCT Skills</Title>}
      extra={
        <Space>
          <Input
            allowClear
            placeholder="Search title / description"
            prefix={<SearchOutlined />}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onPressEnter={() => fetchSkills(search)}
            style={{ width: 260 }}
          />
          <Button onClick={() => fetchSkills(search)} icon={<ReloadOutlined />}>
            Refresh
          </Button>
          {isAdmin ? (
            <Button type="primary" onClick={() => setUploadOpen(true)}>
              Upload ZIP
            </Button>
          ) : null}
        </Space>
      }
    >
      <Table
        rowKey="skill_id"
        loading={loading}
        dataSource={rows}
        columns={columns as any}
        pagination={{ pageSize: 20, showSizeChanger: false }}
      />

      <Drawer
        title={drawerSkill?.display_title ?? "Skill"}
        width={640}
        open={drawerSkill !== null}
        onClose={() => setDrawerSkill(null)}
      >
        {drawerSkill && (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Paragraph type="secondary">{drawerSkill.description}</Paragraph>

            <div>
              <Text strong>Metadata</Text>
              <pre
                style={{
                  background: "#fafafa",
                  padding: 12,
                  marginTop: 8,
                  borderRadius: 6,
                  maxHeight: 200,
                  overflow: "auto",
                }}
              >
                {JSON.stringify(drawerSkill.xct_metadata ?? {}, null, 2)}
              </pre>
            </div>

            {drawerSkill.system_prompt_template ? (
              <div>
                <Text strong>system_prompt_template</Text>
                <pre
                  style={{
                    background: "#fafafa",
                    padding: 12,
                    marginTop: 8,
                    borderRadius: 6,
                    maxHeight: 280,
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {drawerSkill.system_prompt_template}
                </pre>
              </div>
            ) : null}

            {drawerSkill.tool_schema && drawerSkill.tool_schema.length > 0 ? (
              <div>
                <Text strong>tool_schema</Text>
                <pre
                  style={{
                    background: "#fafafa",
                    padding: 12,
                    marginTop: 8,
                    borderRadius: 6,
                    maxHeight: 280,
                    overflow: "auto",
                  }}
                >
                  {JSON.stringify(drawerSkill.tool_schema, null, 2)}
                </pre>
              </div>
            ) : null}
          </Space>
        )}
      </Drawer>

      <Modal
        title="Upload skill ZIP"
        open={uploadOpen}
        footer={null}
        onCancel={() => setUploadOpen(false)}
        destroyOnClose
      >
        <SkillUploadForm
          accessToken={accessToken}
          onSuccess={(s) => {
            setUploadOpen(false);
            message.success(`Uploaded "${s.display_title ?? s.skill_id}"`);
            fetchSkills(search);
          }}
        />
      </Modal>
    </Card>
  );
};

export default SkillTable;
