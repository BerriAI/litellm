import React, { useEffect, useMemo } from "react";
import { Modal, Form, Input, Typography, Tag, Button } from "antd";
import { MCPServer } from "./types";
import {
  HeaderVariable,
  getAllVariablesFor,
  getUserFieldValues,
  serverKeyFor,
  setUserFieldValues,
} from "./header_variables_prototype";
import NotificationsManager from "../molecules/notifications_manager";

const { Text: AntdText, Title: AntdTitle, Paragraph: AntdParagraph } = Typography;

interface MCPUserFieldsModalProps {
  server: MCPServer | null;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

const MCPUserFieldsModal: React.FC<MCPUserFieldsModalProps> = ({ server, open, onClose, onSaved }) => {
  const [form] = Form.useForm();

  const variables: HeaderVariable[] = useMemo(() => {
    if (!server) return [];
    return getAllVariablesFor(server);
  }, [server, open]);

  const perUserVars = variables.filter((v) => v.scope === "per_user");
  const globalVars = variables.filter((v) => v.scope === "global");

  useEffect(() => {
    if (!server || !open) return;
    const key = serverKeyFor(server);
    const altKey = server.alias && server.alias !== key ? server.alias : "";
    const existing = { ...getUserFieldValues(altKey), ...getUserFieldValues(key) };
    form.setFieldsValue(existing);
  }, [server, open, form]);

  const handleSave = async () => {
    if (!server) return;
    try {
      const values = await form.validateFields();
      setUserFieldValues(serverKeyFor(server), values);
      if (server.alias && server.alias !== serverKeyFor(server)) {
        setUserFieldValues(server.alias, values);
      }
      NotificationsManager.success("Your fields were saved. You can now use this MCP server.");
      onSaved();
      onClose();
    } catch {
      // validation errors render inline
    }
  };

  if (!server) return null;

  return (
    <Modal
      open={open}
      title={
        <div className="pb-3 border-b border-gray-100">
          <AntdTitle level={4} className="!mb-1">
            Connect to {server.alias || server.server_name}
          </AntdTitle>
          <AntdText type="secondary" className="text-sm">
            This MCP server needs some user-specific values before you can use it.
          </AntdText>
        </div>
      }
      onCancel={onClose}
      footer={null}
      width={620}
      destroyOnClose
    >
      {perUserVars.length === 0 ? (
        <div className="text-center py-8">
          <AntdText type="secondary">This MCP server has no per-user fields configured.</AntdText>
        </div>
      ) : (
        <Form layout="vertical" form={form} onFinish={handleSave} className="pt-4">
          <AntdParagraph className="text-sm text-gray-600 mb-4">
            These values get interpolated into the request headers when you (and only you) call this
            MCP server. They&apos;re never shared with other users.
          </AntdParagraph>

          {perUserVars.map((v) => (
            <Form.Item
              key={v.name}
              label={
                <span className="text-sm font-medium text-gray-700">
                  <code className="bg-pink-50 text-pink-700 px-1.5 py-0.5 rounded font-mono text-xs">
                    {"${"}
                    {v.name}
                    {"}"}
                  </code>
                </span>
              }
              name={v.name}
              rules={[{ required: true, message: `${v.name} is required` }]}
            >
              <Input.Password
                size="large"
                placeholder={`Your ${v.name.toLowerCase().replace(/_/g, " ")}`}
                autoComplete="off"
              />
            </Form.Item>
          ))}

          {globalVars.length > 0 && (
            <div className="mt-2 mb-6 p-3 bg-gray-50 rounded-lg border border-gray-200">
              <AntdText className="text-xs uppercase tracking-wide text-gray-500 font-medium block mb-2">
                Already set by admin (global)
              </AntdText>
              <div className="flex flex-wrap gap-1.5">
                {globalVars.map((v) => (
                  <Tag key={v.name} color="default" className="font-mono text-xs">
                    {"${"}
                    {v.name}
                    {"}"}
                  </Tag>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-4 border-t border-gray-100">
            <Button onClick={onClose}>Cancel</Button>
            <Button type="primary" htmlType="submit">
              Save and Connect
            </Button>
          </div>
        </Form>
      )}
    </Modal>
  );
};

export default MCPUserFieldsModal;
