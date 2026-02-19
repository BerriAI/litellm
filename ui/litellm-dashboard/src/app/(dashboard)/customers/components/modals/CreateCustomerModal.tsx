import React, { useEffect, useState } from "react";
import { Modal, Form, Input } from "antd";
import { Accordion, AccordionBody, AccordionHeader } from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Tooltip } from "antd";
import type { NewCustomerData } from "@/app/(dashboard)/customers/types";
import CustomerFormFields from "@/app/(dashboard)/customers/components/CustomerFormFields";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import AgentSelector from "@/components/agent_management/AgentSelector";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { customerCreateCall, fetchMCPAccessGroups } from "@/components/networking";

interface CreateCustomerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (customer?: NewCustomerData) => void;
}

const CreateCustomerModal: React.FC<CreateCustomerModalProps> = ({
  isOpen,
  onClose,
  onCreate,
}) => {
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadMCPAccessGroups = async () => {
      if (!accessToken) return;
      try {
        await fetchMCPAccessGroups(accessToken);
      } catch (error) {
        console.error("Failed to fetch MCP access groups:", error);
      }
    };
    if (isOpen) loadMCPAccessGroups();
  }, [accessToken, isOpen]);

  const buildObjectPermission = (values: Record<string, any>) => {
    const objPerm: {
      mcp_servers?: string[];
      mcp_access_groups?: string[];
      mcp_tool_permissions?: Record<string, string[]>;
      agents?: string[];
      agent_access_groups?: string[];
    } = {};
    const mcp = values.allowed_mcp_servers_and_groups;
    if (mcp && (mcp.servers?.length > 0 || mcp.accessGroups?.length > 0)) {
      if (mcp.servers?.length) objPerm.mcp_servers = mcp.servers;
      if (mcp.accessGroups?.length) objPerm.mcp_access_groups = mcp.accessGroups;
    }
    if (values.mcp_tool_permissions && Object.keys(values.mcp_tool_permissions).length > 0) {
      objPerm.mcp_tool_permissions = values.mcp_tool_permissions;
    }
    const agents = values.allowed_agents_and_groups;
    if (agents && (agents.agents?.length > 0 || agents.accessGroups?.length > 0)) {
      if (agents.agents?.length) objPerm.agents = agents.agents;
      if (agents.accessGroups?.length) objPerm.agent_access_groups = agents.accessGroups;
    }
    return Object.keys(objPerm).length > 0 ? objPerm : undefined;
  };

  const handleOk = async () => {
    if (!accessToken) return;
    try {
      const values = await form.validateFields();
      setLoading(true);
      const { allowed_mcp_servers_and_groups, allowed_agents_and_groups, mcp_tool_permissions, ...rest } = values;
      const payload: any = { ...rest };
      const object_permission = buildObjectPermission(values);
      if (object_permission) payload.object_permission = object_permission;
      await customerCreateCall(accessToken, payload);
      form.resetFields();
      onCreate();
      onClose();
    } catch (error) {
      console.error("Validation failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Create New Customer"
      open={isOpen}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={loading}
      okText="Create Customer"
      width={600}
    >
      <Form form={form} layout="vertical" className="mt-4">
        <Form.Item
          label="User ID"
          name="user_id"
          rules={[{ required: true, message: "Please enter a user ID" }]}
        >
          <Input placeholder="e.g. customer-007" />
        </Form.Item>

        <CustomerFormFields form={form} mode="create" />

        <Form.Item label="Metadata (JSON)" name="metadata" initialValue="{}">
          <Input.TextArea rows={3} className="font-mono" placeholder="{}" />
        </Form.Item>

        <Accordion className="mt-4 mb-4">
          <AccordionHeader>
            <b>MCP Settings</b>
          </AccordionHeader>
          <AccordionBody>
            <Form.Item
              label={
                <span>
                  Allowed MCP Servers{" "}
                  <Tooltip title="Select which MCP servers or access groups this customer can access">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="allowed_mcp_servers_and_groups"
              initialValue={{ servers: [], accessGroups: [] }}
              className="mt-4"
              help="Select MCP servers or access groups this customer can access"
            >
              <MCPServerSelector
                onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                accessToken={accessToken || ""}
                placeholder="Select MCP servers or access groups (optional)"
              />
            </Form.Item>
            <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
              <Input type="hidden" />
            </Form.Item>
            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) =>
                prevValues.allowed_mcp_servers_and_groups !== currentValues.allowed_mcp_servers_and_groups ||
                prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
              }
            >
              {() => (
                <div className="mt-6">
                  <MCPToolPermissions
                    accessToken={accessToken || ""}
                    selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []}
                    toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                    onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                  />
                </div>
              )}
            </Form.Item>
          </AccordionBody>
        </Accordion>

        <Accordion className="mt-4 mb-4">
          <AccordionHeader>
            <b>Agent Settings</b>
          </AccordionHeader>
          <AccordionBody>
            <Form.Item
              label={
                <span>
                  Allowed Agents{" "}
                  <Tooltip title="Select which agents or access groups this customer can access">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="allowed_agents_and_groups"
              initialValue={{ agents: [], accessGroups: [] }}
              className="mt-4"
              help="Select agents or access groups this customer can access"
            >
              <AgentSelector
                onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                value={form.getFieldValue("allowed_agents_and_groups")}
                accessToken={accessToken || ""}
                placeholder="Select agents or access groups (optional)"
              />
            </Form.Item>
          </AccordionBody>
        </Accordion>
      </Form>
    </Modal>
  );
};

export default CreateCustomerModal;
