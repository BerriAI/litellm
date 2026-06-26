import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import type { FormInstance } from "antd";
import { Form, Input, Select, Space, Tabs } from "antd";
import { BotIcon, InfoIcon, LayersIcon, ServerIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

const { TextArea } = Input;

export interface AccessGroupFormValues {
  name: string;
  description: string;
  modelIds: string[];
  mcpServerIds: string[];
  agentIds: string[];
}

interface AccessGroupBaseFormProps {
  form: FormInstance<AccessGroupFormValues>;
  isNameDisabled?: boolean;
}

export function AccessGroupBaseForm({ form, isNameDisabled = false }: AccessGroupBaseFormProps) {
  const { t } = useTranslation();
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
          {t("accessGroups.accessGroupBaseForm.tabGeneralInfo")}
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item
            name="name"
            label={t("accessGroups.accessGroupBaseForm.labelGroupName")}
            rules={[
              {
                required: true,
                message: t("accessGroups.accessGroupBaseForm.nameRequired"),
              },
            ]}
          >
            <Input placeholder={t("accessGroups.accessGroupBaseForm.namePlaceholder")} disabled={isNameDisabled} />
          </Form.Item>
          <Form.Item name="description" label={t("common.description")}>
            <TextArea rows={4} placeholder={t("accessGroups.accessGroupBaseForm.descriptionPlaceholder")} />
          </Form.Item>
        </div>
      ),
    },
    {
      key: "2",
      label: (
        <Space align="center" size={4}>
          <LayersIcon size={16} />
          {t("accessGroups.accessGroupBaseForm.tabModels")}
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="modelIds" label={t("accessGroups.accessGroupBaseForm.labelAllowedModels")}>
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
          {t("accessGroups.accessGroupBaseForm.tabMcpServers")}
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="mcpServerIds" label={t("accessGroups.accessGroupBaseForm.labelAllowedMcpServers")}>
            <Select
              mode="multiple"
              placeholder={t("accessGroups.accessGroupBaseForm.mcpServersPlaceholder")}
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
          {t("accessGroups.accessGroupBaseForm.tabAgents")}
        </Space>
      ),
      children: (
        <div style={{ paddingTop: 16 }}>
          <Form.Item name="agentIds" label={t("accessGroups.accessGroupBaseForm.labelAllowedAgents")}>
            <Select
              mode="multiple"
              placeholder={t("accessGroups.accessGroupBaseForm.agentsPlaceholder")}
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
      }}
    >
      <Tabs defaultActiveKey="1" items={items} />
    </Form>
  );
}
