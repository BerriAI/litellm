import React, { useEffect, useState } from "react";
import { Modal, Input, Typography, Tabs, Form } from "antd";
import { CheckCircleFilled, ExclamationCircleFilled, CopyOutlined } from "@ant-design/icons";
import { Button } from "@tremor/react";
import {
  UserField,
  getUserFieldDefs,
  getUserFieldValues,
  setUserFieldValues,
} from "./userFields";
import { MCPServer } from "./types";
import NotificationsManager from "../molecules/notifications_manager";

const { Title, Paragraph, Text } = Typography;

interface UserFieldsModalProps {
  server: MCPServer | null;
  userId: string;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

const UserFieldsModal: React.FC<UserFieldsModalProps> = ({
  server,
  userId,
  open,
  onClose,
  onSaved,
}) => {
  const [defs, setDefs] = useState<UserField[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState<string>("configure");

  useEffect(() => {
    if (!open || !server) return;
    const loadedDefs = getUserFieldDefs(server.server_id);
    setDefs(loadedDefs);
    setValues(getUserFieldValues(server.server_id, userId));
    setActiveTab("configure");
  }, [open, server, userId]);

  if (!server) return null;

  const isFieldMissing = (name: string): boolean =>
    !values[name] ||
    typeof values[name] !== "string" ||
    values[name].trim() === "";

  const missing = defs.filter((f) => isFieldMissing(f.name));
  const allFilled = defs.length > 0 && missing.length === 0;

  const serverDisplayName = server.server_name || server.alias || server.server_id;

  const handleSave = () => {
    setUserFieldValues(server.server_id, userId, values);
    NotificationsManager.success(
      allFilled
        ? `Saved. ${serverDisplayName} is ready to use.`
        : `Saved ${defs.length - missing.length} of ${defs.length} fields.`,
    );
    onSaved();
    if (allFilled) {
      onClose();
    }
  };

  const buildDeepLinkUrl = (): string => {
    if (typeof window === "undefined") {
      return `<dashboard-url>?page=mcp-servers&openUserFields=${server.server_id}`;
    }
    const params = new URLSearchParams();
    params.set("page", "mcp-servers");
    params.set("openUserFields", server.server_id);
    return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
  };

  const previewStatusLine =
    defs.length === 0
      ? "No fields defined."
      : missing.length === 0
        ? "All fields configured."
        : `Missing field${missing.length === 1 ? "" : "s"}:`;

  const errorPreview = `Error: MCP server "${serverDisplayName}" requires user configuration before use.

${previewStatusLine}
${missing.map((f) => `  • ${f.label || f.name}${f.description ? ` — ${f.description}` : ""}`).join("\n")}

Please configure your fields at:
${buildDeepLinkUrl()}

Once configured, retry your request.`;

  return (
    <Modal
      open={open}
      onCancel={onClose}
      width={680}
      footer={null}
      title={
        <div className="flex items-center gap-2 pb-2 border-b border-gray-100">
          {allFilled ? (
            <CheckCircleFilled style={{ color: "#10b981", fontSize: 22 }} />
          ) : (
            <ExclamationCircleFilled style={{ color: "#ef4444", fontSize: 22 }} />
          )}
          <Title level={4} style={{ margin: 0 }}>
            {allFilled ? "Configured: " : "Configure your fields for "}
            {serverDisplayName}
          </Title>
        </div>
      }
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: "configure",
            label: "Your Fields",
            children: (
              <div className="pt-2">
                <Paragraph type="secondary">
                  This MCP server requires per-user configuration. Fill in the fields below to start
                  using it.
                </Paragraph>

                {defs.length === 0 ? (
                  <div className="py-6 text-center text-gray-400">
                    No per-user fields are defined for this server.
                  </div>
                ) : (
                  <Form layout="vertical">
                    {defs.map((f) => {
                      const isMissing = isFieldMissing(f.name);
                      return (
                        <Form.Item
                          key={f.name}
                          label={
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{f.label || f.name}</span>
                              {isMissing ? (
                                <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
                                  required
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200">
                                  <CheckCircleFilled style={{ fontSize: 10 }} /> set
                                </span>
                              )}
                            </div>
                          }
                          help={f.description}
                        >
                          {f.secret ? (
                            <Input.Password
                              placeholder={`Enter your ${f.label || f.name}`}
                              value={values[f.name] || ""}
                              onChange={(e) =>
                                setValues({ ...values, [f.name]: e.target.value })
                              }
                            />
                          ) : (
                            <Input
                              placeholder={`Enter your ${f.label || f.name}`}
                              value={values[f.name] || ""}
                              onChange={(e) =>
                                setValues({ ...values, [f.name]: e.target.value })
                              }
                            />
                          )}
                        </Form.Item>
                      );
                    })}
                  </Form>
                )}

                <div className="flex justify-end gap-2 pt-3 border-t border-gray-100 mt-3">
                  <Button variant="secondary" onClick={onClose}>
                    Cancel
                  </Button>
                  <Button variant="primary" onClick={handleSave} disabled={defs.length === 0}>
                    Save
                  </Button>
                </div>
              </div>
            ),
          },
          {
            key: "preview",
            label: "Preview Claude Code Error",
            children: (
              <div className="pt-2">
                <Paragraph type="secondary">
                  This is what your users will see in their terminal when they try to use{" "}
                  <Text code>{serverDisplayName}</Text> via Claude Code without configuring these
                  fields:
                </Paragraph>
                <div
                  className="rounded-lg bg-gray-900 text-gray-100 font-mono text-xs p-4 whitespace-pre-wrap relative"
                  style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
                >
                  <button
                    onClick={() => {
                      if (typeof navigator !== "undefined" && navigator.clipboard) {
                        navigator.clipboard.writeText(errorPreview);
                        NotificationsManager.success("Copied error preview");
                      }
                    }}
                    className="absolute top-2 right-2 text-gray-400 hover:text-white p-1"
                  >
                    <CopyOutlined />
                  </button>
                  {errorPreview}
                </div>
                <Paragraph type="secondary" style={{ marginTop: 12, fontSize: 12 }}>
                  The link in the error opens this same dialog directly so users can fix and retry in
                  one click.
                </Paragraph>
              </div>
            ),
          },
        ]}
      />
    </Modal>
  );
};

export default UserFieldsModal;
