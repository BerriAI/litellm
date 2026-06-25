import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import PassThroughRoutesSelector from "@/components/common_components/PassThroughRoutesSelector";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import type { FormInstance } from "antd";
import { Form, Input, Select, Space, Tabs } from "antd";
import { BotIcon, DatabaseIcon, InfoIcon, LayersIcon, RouteIcon, ServerIcon } from "lucide-react";

const { TextArea } = Input;

export interface AccessGroupFormValues {
  name: string;
  description: string;
  modelIds: string[];
  mcpServerIds: string[];
  agentIds: string[];
  passthroughRoutes: string[];
  vectorStoreIds: string[];
}

interface AccessGroupBaseFormProps {
  form: FormInstance<AccessGroupFormValues>;
  accessToken: string;
  isNameDisabled?: boolean;
}

export function AccessGroupBaseForm({ form, accessToken, isNameDisabled = false }: AccessGroupBaseFormProps) {
  const { data: agentsData } = useAgents();
  const { data: mcpServersData } = useMCPServers();

  const agents = agentsData?.agents ?? [];
  const mcpServers = mcpServersData ?? [];
  const items = [
    {
      key: "1",
      label: (
        <Space align="center" size={4}>
          <InfoIcon size={16} />
          General Info
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item
            name="name"
            label="Group Name"
            rules={[
              {
                required: true,
                message: "Please enter the access group name",
              },
            ]}
          >
            <Input placeholder="e.g. Engineering Team" disabled={isNameDisabled} />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <TextArea rows={4} placeholder="Describe the purpose of this access group..." />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "2",
      label: (
        <Space align="center" size={4}>
          <LayersIcon size={16} />
          Models
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="modelIds" label="Allowed Models">
            <ModelSelect
              context="global"
              value={form.getFieldValue("modelIds") ?? []}
              onChange={(values) => form.setFieldsValue({ modelIds: values })}
              style={{ width: "100%" }}
            />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "3",
      label: (
        <Space align="center" size={4}>
          <ServerIcon size={16} />
          MCP Servers
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="mcpServerIds" label="Allowed MCP Servers">
            <Select
              mode="multiple"
              placeholder="Select MCP servers"
              style={{ width: "100%" }}
              optionFilterProp="label"
              allowClear
              options={mcpServers.map((server) => ({
                label: server.server_name ?? server.server_id,
                value: server.server_id,
              }))}
            />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "4",
      label: (
        <Space align="center" size={4}>
          <BotIcon size={16} />
          Agents
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="agentIds" label="Allowed Agents">
            <Select
              mode="multiple"
              placeholder="Select agents"
              style={{ width: "100%" }}
              optionFilterProp="label"
              allowClear
              options={agents.map((agent) => ({
                label: agent.agent_name,
                value: agent.agent_id,
              }))}
            />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "5",
      label: (
        <Space align="center" size={4}>
          <RouteIcon size={16} />
          Pass Through Routes
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="passthroughRoutes" label="Allowed Pass Through Routes">
            <PassThroughRoutesSelector
              accessToken={accessToken}
              value={form.getFieldValue("passthroughRoutes") ?? []}
              onChange={(values) => form.setFieldsValue({ passthroughRoutes: values })}
            />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "6",
      label: (
        <Space align="center" size={4}>
          <DatabaseIcon size={16} />
          Vector Stores
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="vectorStoreIds" label="Allowed Vector Stores">
            <VectorStoreSelector
              accessToken={accessToken}
              value={form.getFieldValue("vectorStoreIds") ?? []}
              onChange={(values) => form.setFieldsValue({ vectorStoreIds: values })}
            />
          </Form.Item>
        </div>
      ),
    },
  ];

  return (
    <Form
      form={form}
      layout="vertical"
      name="access_group_form"
      initialValues={{
        modelIds: [],
        mcpServerIds: [],
        agentIds: [],
        passthroughRoutes: [],
        vectorStoreIds: [],
      }}
    >
      <Tabs defaultActiveKey="1" items={items} />
    </Form>
  );
}
