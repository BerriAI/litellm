import React, { useState, useEffect } from "react";
import { Card, Title } from "@tremor/react";
import { Button as TremorButton } from "@tremor/react";
import { Form, Input, Button as AntButton, message, Spin, Collapse, Switch, Descriptions } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import { getAgentInfo, patchAgentCall } from "../networking";
import { Agent } from "./types";

const { Panel } = Collapse;

interface AgentInfoViewProps {
  agentId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const AgentInfoView: React.FC<AgentInfoViewProps> = ({
  agentId,
  onClose,
  accessToken,
  isAdmin,
}) => {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchAgentInfo();
  }, [agentId, accessToken]);

  const fetchAgentInfo = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const data = await getAgentInfo(accessToken, agentId);
      setAgent(data);
      
      // Parse skills to handle tags and examples as comma-separated strings for form
      const skills = data.agent_card_params?.skills?.map((skill: any) => ({
        ...skill,
        tags: skill.tags,
        examples: skill.examples || [],
      })) || [];

      form.setFieldsValue({
        agent_name: data.agent_name,
        name: data.agent_card_params?.name,
        description: data.agent_card_params?.description,
        url: data.agent_card_params?.url,
        version: data.agent_card_params?.version,
        protocolVersion: data.agent_card_params?.protocolVersion,
        streaming: data.agent_card_params?.capabilities?.streaming,
        pushNotifications: data.agent_card_params?.capabilities?.pushNotifications,
        stateTransitionHistory: data.agent_card_params?.capabilities?.stateTransitionHistory,
        skills: skills,
        iconUrl: data.agent_card_params?.iconUrl,
        documentationUrl: data.agent_card_params?.documentationUrl,
        supportsAuthenticatedExtendedCard: data.agent_card_params?.supportsAuthenticatedExtendedCard,
        model: data.litellm_params?.model,
        make_public: data.litellm_params?.make_public,
      });
    } catch (error) {
      console.error("Error fetching agent info:", error);
      message.error("Failed to load agent information");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdate = async (values: any) => {
    if (!accessToken || !agent) return;

    setIsSaving(true);
    try {
      const updateData = {
        agent_name: values.agent_name,
        agent_card_params: {
          protocolVersion: values.protocolVersion || "1.0",
          name: values.name,
          description: values.description,
          url: values.url,
          version: values.version || "1.0.0",
          defaultInputModes: agent.agent_card_params?.defaultInputModes || ["text"],
          defaultOutputModes: agent.agent_card_params?.defaultOutputModes || ["text"],
          capabilities: {
            streaming: values.streaming !== false,
            ...(values.pushNotifications !== undefined && { pushNotifications: values.pushNotifications }),
            ...(values.stateTransitionHistory !== undefined && { stateTransitionHistory: values.stateTransitionHistory }),
          },
          skills: values.skills || [],
          ...(values.iconUrl && { iconUrl: values.iconUrl }),
          ...(values.documentationUrl && { documentationUrl: values.documentationUrl }),
          ...(values.supportsAuthenticatedExtendedCard !== undefined && { 
            supportsAuthenticatedExtendedCard: values.supportsAuthenticatedExtendedCard 
          }),
        },
        ...(values.model || values.make_public !== undefined ? {
          litellm_params: {
            ...(values.model && { model: values.model }),
            ...(values.make_public !== undefined && { make_public: values.make_public }),
          }
        } : {}),
      };

      await patchAgentCall(accessToken, agentId, updateData);
      message.success("Agent updated successfully");
      setIsEditing(false);
      fetchAgentInfo();
    } catch (error) {
      console.error("Error updating agent:", error);
      message.error("Failed to update agent");
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <div className="flex justify-center items-center h-64">
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (!agent) {
    return (
      <Card>
        <div className="text-center">Agent not found</div>
        <TremorButton onClick={onClose} className="mt-4">
          Back to Agents List
        </TremorButton>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>Agent Details</Title>
        <TremorButton onClick={onClose} variant="secondary">
          Back to List
        </TremorButton>
      </div>

      {!isEditing ? (
        <div>
          <Descriptions bordered column={1}>
            <Descriptions.Item label="Agent ID">{agent.agent_id}</Descriptions.Item>
            <Descriptions.Item label="Agent Name">{agent.agent_name}</Descriptions.Item>
            <Descriptions.Item label="Display Name">{agent.agent_card_params?.name}</Descriptions.Item>
            <Descriptions.Item label="Description">{agent.agent_card_params?.description}</Descriptions.Item>
            <Descriptions.Item label="URL">{agent.agent_card_params?.url}</Descriptions.Item>
            <Descriptions.Item label="Version">{agent.agent_card_params?.version}</Descriptions.Item>
            <Descriptions.Item label="Protocol Version">{agent.agent_card_params?.protocolVersion}</Descriptions.Item>
            <Descriptions.Item label="Streaming">{agent.agent_card_params?.capabilities?.streaming ? 'Yes' : 'No'}</Descriptions.Item>
            <Descriptions.Item label="Skills">
              {agent.agent_card_params?.skills?.length || 0} skills configured
            </Descriptions.Item>
            {agent.litellm_params?.model && (
              <Descriptions.Item label="Model">{agent.litellm_params.model}</Descriptions.Item>
            )}
            {agent.created_at && (
              <Descriptions.Item label="Created At">{new Date(agent.created_at).toLocaleString()}</Descriptions.Item>
            )}
            {agent.updated_at && (
              <Descriptions.Item label="Updated At">{new Date(agent.updated_at).toLocaleString()}</Descriptions.Item>
            )}
          </Descriptions>

          {isAdmin && (
            <div style={{ marginTop: 16 }}>
              <AntButton type="primary" onClick={() => setIsEditing(true)}>
                Edit Agent
              </AntButton>
            </div>
          )}
        </div>
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpdate}
        >
          <Form.Item label="Agent ID">
            <Input value={agent.agent_id} disabled />
          </Form.Item>

          <Form.Item
            label="Agent Name"
            name="agent_name"
            rules={[{ required: true, message: "Please enter an agent name" }]}
          >
            <Input />
          </Form.Item>

          <Collapse defaultActiveKey={['basic']} style={{ marginBottom: 16 }}>
            <Panel header="Basic Information" key="basic">
              <Form.Item
                label="Display Name"
                name="name"
                rules={[{ required: true, message: "Required" }]}
              >
                <Input />
              </Form.Item>

              <Form.Item
                label="Description"
                name="description"
                rules={[{ required: true, message: "Required" }]}
              >
                <Input.TextArea rows={3} />
              </Form.Item>

              <Form.Item
                label="URL"
                name="url"
                rules={[{ required: true, message: "Required" }]}
              >
                <Input />
              </Form.Item>

              <Form.Item label="Version" name="version">
                <Input />
              </Form.Item>

              <Form.Item label="Protocol Version" name="protocolVersion">
                <Input />
              </Form.Item>
            </Panel>

            <Panel header="Skills" key="skills">
              <Form.List name="skills">
                {(fields, { add, remove }) => (
                  <>
                    {fields.map((field, index) => (
                      <div key={field.key} style={{ marginBottom: 16, padding: 16, border: '1px solid #d9d9d9', borderRadius: 4 }}>
                        <Form.Item
                          {...field}
                          label="Skill ID"
                          name={[field.name, 'id']}
                          rules={[{ required: true, message: 'Required' }]}
                        >
                          <Input />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="Skill Name"
                          name={[field.name, 'name']}
                          rules={[{ required: true, message: 'Required' }]}
                        >
                          <Input />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="Description"
                          name={[field.name, 'description']}
                          rules={[{ required: true, message: 'Required' }]}
                        >
                          <Input.TextArea rows={2} />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="Tags (comma-separated)"
                          name={[field.name, 'tags']}
                          rules={[{ required: true, message: 'Required' }]}
                          getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim())}
                          getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : value })}
                        >
                          <Input />
                        </Form.Item>
                        <Form.Item
                          {...field}
                          label="Examples (comma-separated)"
                          name={[field.name, 'examples']}
                          getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim()).filter((s: string) => s)}
                          getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : '' })}
                        >
                          <Input />
                        </Form.Item>
                        <AntButton 
                          type="link" 
                          danger 
                          onClick={() => remove(field.name)}
                          icon={<MinusCircleOutlined />}
                        >
                          Remove Skill
                        </AntButton>
                      </div>
                    ))}
                    <AntButton 
                      type="dashed" 
                      onClick={() => add()} 
                      icon={<PlusOutlined />}
                      style={{ width: '100%' }}
                    >
                      Add Skill
                    </AntButton>
                  </>
                )}
              </Form.List>
            </Panel>

            <Panel header="Capabilities" key="capabilities">
              <Form.Item
                label="Streaming"
                name="streaming"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                label="Push Notifications"
                name="pushNotifications"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                label="State Transition History"
                name="stateTransitionHistory"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Panel>

            <Panel header="Optional Settings" key="optional">
              <Form.Item label="Icon URL" name="iconUrl">
                <Input />
              </Form.Item>

              <Form.Item label="Documentation URL" name="documentationUrl">
                <Input />
              </Form.Item>

              <Form.Item
                label="Supports Authenticated Extended Card"
                name="supportsAuthenticatedExtendedCard"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Panel>

            <Panel header="LiteLLM Parameters" key="litellm">
              <Form.Item label="Model (Optional)" name="model">
                <Input />
              </Form.Item>

              <Form.Item
                label="Make Public"
                name="make_public"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Panel>
          </Collapse>

          <Form.Item>
            <div style={{ display: "flex", gap: "8px" }}>
              <AntButton
                type="primary"
                htmlType="submit"
                loading={isSaving}
              >
                Save Changes
              </AntButton>
              <AntButton onClick={() => {
                setIsEditing(false);
                fetchAgentInfo();
              }}>
                Cancel
              </AntButton>
            </div>
          </Form.Item>
        </Form>
      )}
    </Card>
  );
};

export default AgentInfoView;

