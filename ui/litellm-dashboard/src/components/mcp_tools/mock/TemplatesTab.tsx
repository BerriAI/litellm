// PROTOTYPE: list view of MCP server templates. Cards mirror the look of
// MCPServerCard but stripped down to template-only metadata.

import React, { useEffect, useState } from "react";
import {
  Button,
  Card,
  Empty,
  Input,
  Modal,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  MCPTemplate,
  deleteTemplate,
  listTemplates,
  subscribeTemplatesChanged,
} from "./mockTemplates";
import CreateMCPServer from "../create_mcp_server";
import MCPDiscovery from "../mcp_discovery";
import { DiscoverableMCPServer } from "../types";

const { Title, Text, Paragraph } = Typography;

interface TemplatesTabProps {
  isAdmin: boolean;
  userRole: string;
  accessToken: string | null;
  availableAccessGroups: string[];
}

const TemplatesTab: React.FC<TemplatesTabProps> = ({
  isAdmin,
  userRole,
  accessToken,
  availableAccessGroups,
}) => {
  const [templates, setTemplates] = useState<MCPTemplate[]>([]);
  const [search, setSearch] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<MCPTemplate | null>(null);
  const [deleting, setDeleting] = useState<MCPTemplate | null>(null);
  // PROTOTYPE: discovery picker before the form, mirroring the From Blank
  // path on the Instances page. Picking a preset prefills the template form;
  // Custom Server opens it blank. Editing skips discovery entirely.
  const [discoveryOpen, setDiscoveryOpen] = useState(false);
  const [prefillData, setPrefillData] =
    useState<DiscoverableMCPServer | null>(null);

  const refresh = () => setTemplates(listTemplates());

  useEffect(() => {
    refresh();
    return subscribeTemplatesChanged(refresh);
  }, []);

  const filtered = templates.filter((t) => {
    const q = search.trim().toLowerCase();
    if (!q) return true;
    return (
      t.name.toLowerCase().includes(q) ||
      (t.description ?? "").toLowerCase().includes(q) ||
      (t.url ?? "").toLowerCase().includes(q)
    );
  });

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-3">
            <Title level={4} style={{ margin: 0 }}>
              Templates
            </Title>
            <Tag color="purple">Prototype</Tag>
            {templates.length > 0 && (
              <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200">
                {templates.length}
              </span>
            )}
          </div>
          <Text type="secondary" className="text-sm">
            Reusable blueprints for MCP servers. Define structure once, then
            spin up instances with per-instance and per-user variables.
          </Text>
        </div>
        {isAdmin && (
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setEditing(null);
              setPrefillData(null);
              setDiscoveryOpen(true);
            }}
          >
            New Template
          </Button>
        )}
      </div>

      <Input
        allowClear
        prefix={<SearchOutlined className="text-gray-400" />}
        placeholder="Search templates by name, description, or URL"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{ maxWidth: 360 }}
        className="mb-4"
      />

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-200 bg-white p-12">
          <Empty
            description={
              templates.length === 0
                ? "No templates yet. Click 'New Template' to create your first one."
                : "No templates match your search."
            }
          />
        </div>
      ) : (
        <div className="grid auto-rows-fr grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((t) => {
            const perUserCount = t.variables.filter(
              (v) => v.scope === "per_user",
            ).length;
            const instanceCount = t.variables.filter(
              (v) => v.scope === "instance",
            ).length;
            return (
              <Card
                key={t.template_id}
                hoverable
                className="border border-gray-200"
                styles={{ body: { padding: 16 } }}
                actions={
                  isAdmin
                    ? [
                        <Tooltip title="Edit" key="edit">
                          <EditOutlined
                            onClick={() => {
                              setEditing(t);
                              setPrefillData(null);
                              setModalOpen(true);
                            }}
                          />
                        </Tooltip>,
                        <Tooltip title="Delete" key="delete">
                          <DeleteOutlined
                            onClick={() => setDeleting(t)}
                            className="hover:text-red-500"
                          />
                        </Tooltip>,
                      ]
                    : undefined
                }
              >
                <div className="flex items-start gap-3 mb-2">
                  <div className="flex-1 min-w-0">
                    <Title
                      level={5}
                      ellipsis
                      style={{ margin: 0 }}
                      className="truncate"
                    >
                      {t.name}
                    </Title>
                    <div className="flex items-center gap-2 mt-1">
                      <Tag color="blue" style={{ marginRight: 0 }}>
                        {t.transport.toUpperCase()}
                      </Tag>
                      {t.auth_type && t.auth_type !== "none" && (
                        <Tag color="default" style={{ marginRight: 0 }}>
                          {t.auth_type}
                        </Tag>
                      )}
                    </div>
                  </div>
                </div>

                {t.description && (
                  <Paragraph
                    ellipsis={{ rows: 2 }}
                    className="text-xs text-gray-600 mb-2"
                  >
                    {t.description}
                  </Paragraph>
                )}

                {t.url && (
                  <div className="mb-2">
                    <Text className="text-xs text-gray-500 block">URL</Text>
                    <Text code className="text-xs break-all">
                      {t.url}
                    </Text>
                  </div>
                )}

                <div className="flex items-center gap-2 flex-wrap mt-2">
                  <Tag color="purple" style={{ marginRight: 0 }}>
                    {t.variables.length} variable
                    {t.variables.length === 1 ? "" : "s"}
                  </Tag>
                  {instanceCount > 0 && (
                    <Tag color="geekblue" style={{ marginRight: 0 }}>
                      {instanceCount} instance-scoped
                    </Tag>
                  )}
                  {perUserCount > 0 && (
                    <Tag color="magenta" style={{ marginRight: 0 }}>
                      {perUserCount} per-user
                    </Tag>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Discovery picker — first page of the New Template flow. Mirrors
          the From Blank path on the Instances page so admins can start a
          template from a preset (Postgres, GitHub, etc.) or Custom Server. */}
      <MCPDiscovery
        isVisible={discoveryOpen}
        onClose={() => setDiscoveryOpen(false)}
        onSelectServer={(server: DiscoverableMCPServer) => {
          setPrefillData(server);
          setDiscoveryOpen(false);
          setModalOpen(true);
        }}
        onCustomServer={() => {
          setPrefillData(null);
          setDiscoveryOpen(false);
          setModalOpen(true);
        }}
        accessToken={accessToken}
      />

      {/* Reuse the full Add-MCP-Server modal in template mode — same fields,
          but the Variables section hides the value column and submission
          saves to mockTemplates instead of calling the create API. */}
      <CreateMCPServer
        mode="template"
        initialTemplate={editing}
        prefillData={prefillData}
        onBackToDiscovery={() => {
          setModalOpen(false);
          setPrefillData(null);
          setDiscoveryOpen(true);
        }}
        userRole={userRole}
        accessToken={accessToken}
        availableAccessGroups={availableAccessGroups}
        isModalVisible={modalOpen}
        setModalVisible={(v) => {
          setModalOpen(v);
          if (!v) {
            setEditing(null);
            setPrefillData(null);
          }
        }}
        onCreateSuccess={() => {
          /* Templates don't create an MCP server; handled via onTemplateSaved. */
        }}
        onTemplateSaved={() => refresh()}
      />

      <Modal
        open={!!deleting}
        title="Delete template?"
        okText="Delete"
        okButtonProps={{ danger: true }}
        onCancel={() => setDeleting(null)}
        onOk={() => {
          if (deleting) deleteTemplate(deleting.template_id);
          setDeleting(null);
        }}
      >
        <Text className="text-gray-600">
          Existing instances created from this template are unaffected, but you
          won&apos;t be able to create new instances from{" "}
          <b>{deleting?.name}</b> after deletion.
        </Text>
      </Modal>
    </div>
  );
};

export default TemplatesTab;
