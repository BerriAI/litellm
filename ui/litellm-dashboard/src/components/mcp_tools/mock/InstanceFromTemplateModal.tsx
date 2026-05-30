// PROTOTYPE: create a new MCP instance from a previously-defined template.
// The user picks a template, fills in only the instance-scoped variable
// values, and gets a server entry created via the existing createMCPServer
// endpoint. Per-user variables are left blank — they come from the user's
// global Variables tab at runtime.

import React, { useEffect, useMemo, useState } from "react";
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  Typography,
  Alert,
  Tag,
  Empty,
} from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { MCPTemplate, listTemplates } from "./mockTemplates";
import { mcpLogoImg } from "../create_mcp_server";
import { createMCPServer } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import { EnvVarDefinition } from "./mockMcpEnvVars";
import { MCPServer } from "../types";

const { Title, Text, Paragraph } = Typography;

interface InstanceFromTemplateModalProps {
  open: boolean;
  onClose: () => void;
  onCreated: (server: MCPServer) => void;
  accessToken: string;
  initialTemplateId?: string | null;
  onBack?: () => void;
}

const InstanceFromTemplateModal: React.FC<InstanceFromTemplateModalProps> = ({
  open,
  onClose,
  onCreated,
  accessToken,
  initialTemplateId,
  onBack,
}) => {
  const [templates, setTemplates] = useState<MCPTemplate[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialTemplateId ?? null,
  );
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open) {
      setTemplates(listTemplates());
      setSelectedId(initialTemplateId ?? null);
      form.resetFields();
    }
  }, [open, initialTemplateId, form]);

  const selected = useMemo(
    () => templates.find((t) => t.template_id === selectedId) ?? null,
    [templates, selectedId],
  );

  const instanceVars = selected?.variables.filter(
    (v) => v.scope === "instance",
  );
  const perUserVars = selected?.variables.filter(
    (v) => v.scope === "per_user",
  );

  const handleCreate = async () => {
    if (!selected || loading) return;
    setLoading(true);
    try {
      const values = await form.validateFields();

      const envVars: EnvVarDefinition[] = selected.variables.map((v) => ({
        name: v.name,
        value: v.scope === "instance" ? (values[`var_${v.name}`] ?? "") : "",
        scope: v.scope,
      }));

      const staticHeaders = (selected.static_headers || []).reduce<
        Record<string, string>
      >((acc, h) => {
        if (h.header) acc[h.header] = h.value ?? "";
        return acc;
      }, {});

      // Build payload from the template's form_snapshot (OAuth URLs, BYOK
      // toggles, allowed_tools, etc.) and overlay the instance-specific
      // fields the user just filled in. Snapshot's variable-scope values
      // are intentionally discarded — env_vars below is the source of truth.
      const {
        mock_env_vars: _ignoreVars,
        ...snapshot
      } = selected.form_snapshot || {};
      const payload: Record<string, unknown> = {
        ...snapshot,
        server_name: values.server_name,
        alias: values.alias || values.server_name,
        description: selected.description,
        url: selected.url,
        transport: selected.transport === "openapi" ? "http" : selected.transport,
        auth_type: selected.auth_type || "none",
        mcp_info: {
          server_name: values.server_name,
          description: selected.description,
          logo_url: selected.logo_url || undefined,
        },
        env_vars: envVars,
        static_headers: staticHeaders,
        // Tag the new instance with the template id (string passed through;
        // backend ignores unknown fields in this prototype path).
        template_id: selected.template_id,
      };

      const response = await createMCPServer(accessToken, payload);
      NotificationsManager.success(
        `Created "${values.server_name}" from template "${selected.name}"`,
      );
      onCreated(response);
      onClose();
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      // antd shows validation errors inline; only surface real API errors.
      if (reason && !reason.toLowerCase().includes("validation")) {
        NotificationsManager.fromBackend(`Error creating instance: ${reason}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={720}
      title={
        <div className="flex items-center gap-3 pb-3 border-b border-gray-100">
          {onBack && (
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={onBack}
              style={{ flexShrink: 0 }}
            />
          )}
          <img
            src={mcpLogoImg}
            alt="MCP"
            style={{ width: 20, height: 20, objectFit: "contain" }}
          />
          <Title level={4} style={{ margin: 0 }}>
            New Instance from Template
          </Title>
          <Tag color="purple">Prototype</Tag>
        </div>
      }
      className="top-8"
    >
      <div className="mt-4 space-y-4">
        <div>
          <Text strong className="text-sm block mb-1">
            Template
          </Text>
          {templates.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-6">
              <Empty
                description={
                  <span className="text-sm">
                    No templates yet. Create one from the{" "}
                    <b>Templates</b> tab first.
                  </span>
                }
              />
            </div>
          ) : (
            <Select
              value={selectedId ?? undefined}
              onChange={(v) => setSelectedId(v)}
              placeholder="Choose a template"
              size="large"
              style={{ width: "100%" }}
              options={templates.map((t) => ({
                value: t.template_id,
                label: (
                  <div className="flex items-center gap-2">
                    <span>{t.name}</span>
                    <Tag color="blue" style={{ marginRight: 0 }}>
                      {t.transport.toUpperCase()}
                    </Tag>
                    <Tag color="purple" style={{ marginRight: 0 }}>
                      {t.variables.length} vars
                    </Tag>
                  </div>
                ),
              }))}
            />
          )}
        </div>

        {selected && (
          <>
            {selected.description && (
              <Paragraph className="text-xs text-gray-600 mb-0">
                {selected.description}
              </Paragraph>
            )}

            <div className="rounded-lg border border-gray-200 p-3 text-xs space-y-1">
              <div>
                <Text className="text-gray-500">URL:</Text>{" "}
                <Text code>{selected.url || "—"}</Text>
              </div>
              <div>
                <Text className="text-gray-500">Auth:</Text>{" "}
                <Text>{selected.auth_type || "none"}</Text>
              </div>
              <div>
                <Text className="text-gray-500">Transport:</Text>{" "}
                <Text>{selected.transport}</Text>
              </div>
            </div>

            <Form form={form} layout="vertical" className="space-y-2">
              <Form.Item
                label={<Text strong className="text-sm">Instance Name</Text>}
                name="server_name"
                rules={[
                  { required: true, message: "Please enter an instance name" },
                ]}
              >
                <Input placeholder="e.g. Postgres_Prod, Postgres_Staging" />
              </Form.Item>
              <Form.Item
                label={<Text strong className="text-sm">Alias (optional)</Text>}
                name="alias"
              >
                <Input placeholder="Defaults to the instance name" />
              </Form.Item>

              {instanceVars && instanceVars.length > 0 && (
                <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 mt-2">
                  <Text strong className="text-sm block mb-2">
                    Instance Variables
                  </Text>
                  <Text className="text-xs text-gray-600 block mb-3">
                    These values are shared by everyone using this instance.
                  </Text>
                  {instanceVars.map((v) => (
                    <Form.Item
                      key={v.name}
                      label={
                        <span className="font-mono text-sm font-semibold">
                          {v.name}
                        </span>
                      }
                      name={`var_${v.name}`}
                      rules={[
                        {
                          required: true,
                          message: `${v.name} is required`,
                        },
                      ]}
                    >
                      <Input
                        placeholder={`Enter value for ${v.name}`}
                        className="font-mono"
                      />
                    </Form.Item>
                  ))}
                </div>
              )}

              {perUserVars && perUserVars.length > 0 && (
                <Alert
                  type="info"
                  showIcon
                  className="mt-2"
                  message={
                    <span className="text-sm">
                      <b>{perUserVars.length}</b> per-user variable
                      {perUserVars.length === 1 ? "" : "s"} (
                      {perUserVars.map((v, i) => (
                        <React.Fragment key={v.name}>
                          {i > 0 ? ", " : ""}
                          <code className="font-mono">{v.name}</code>
                        </React.Fragment>
                      ))}
                      ) will be resolved at runtime from each user&apos;s{" "}
                      <b>Variables</b> tab.
                    </span>
                  }
                />
              )}
            </Form>
          </>
        )}

        <div className="flex items-center justify-end gap-2 pt-4 border-t border-gray-100">
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="primary"
            onClick={handleCreate}
            loading={loading}
            disabled={!selected}
          >
            Create Instance
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default InstanceFromTemplateModal;
