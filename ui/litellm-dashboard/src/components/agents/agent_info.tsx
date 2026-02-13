import React, { useState, useEffect } from "react";
import { Card, Title, Text, Button as TremorButton, Tab, TabGroup, TabList, TabPanel, TabPanels} from "@tremor/react";
import { Form, Input, Button as AntButton, message, Spin, Descriptions } from "antd";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { getAgentInfo, patchAgentCall, getAgentCreateMetadata, AgentCreateInfo } from "../networking";
import { Agent } from "./types";
import AgentFormFields from "./agent_form_fields";
import DynamicAgentFormFields, { buildDynamicAgentData } from "./dynamic_agent_form_fields";
import { buildAgentDataFromForm, parseAgentForForm } from "./agent_config";
import AgentCostView from "./agent_cost_view";
import { detectAgentType, parseDynamicAgentForForm } from "./agent_type_utils";

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
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);
  const [detectedAgentType, setDetectedAgentType] = useState<string>("a2a");

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const metadata = await getAgentCreateMetadata();
        setAgentTypeMetadata(metadata);
      } catch (error) {
        console.error("Error fetching agent metadata:", error);
      }
    };
    fetchMetadata();
  }, []);

  useEffect(() => {
    fetchAgentInfo();
  }, [agentId, accessToken]);

  const fetchAgentInfo = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const data = await getAgentInfo(accessToken, agentId);
      setAgent(data);
      
      // Detect agent type
      const agentType = detectAgentType(data);
      setDetectedAgentType(agentType);
      
      // Parse form values based on agent type
      if (agentType === "a2a") {
        form.setFieldsValue(parseAgentForForm(data));
      } else {
        const typeInfo = agentTypeMetadata.find(t => t.agent_type === agentType);
        if (typeInfo) {
          form.setFieldsValue(parseDynamicAgentForForm(data, typeInfo));
        } else {
      form.setFieldsValue(parseAgentForForm(data));
        }
      }
    } catch (error) {
      console.error("Error fetching agent info:", error);
      message.error("Failed to load agent information");
    } finally {
      setIsLoading(false);
    }
  };

  // Re-parse form when metadata is loaded
  useEffect(() => {
    if (agent && agentTypeMetadata.length > 0) {
      const agentType = detectAgentType(agent);
      if (agentType !== "a2a") {
        const typeInfo = agentTypeMetadata.find(t => t.agent_type === agentType);
        if (typeInfo) {
          form.setFieldsValue(parseDynamicAgentForForm(agent, typeInfo));
        }
      }
    }
  }, [agentTypeMetadata, agent]);

  const selectedAgentTypeInfo = agentTypeMetadata.find(t => t.agent_type === detectedAgentType);

  const handleUpdate = async (values: any) => {
    if (!accessToken || !agent) return;

    setIsSaving(true);
    try {
      let updateData: any;
      
      if (detectedAgentType === "a2a") {
        updateData = buildAgentDataFromForm(values, agent);
      } else if (selectedAgentTypeInfo) {
        updateData = buildDynamicAgentData(values, selectedAgentTypeInfo);
        // Preserve the agent_name from form
        updateData.agent_name = values.agent_name;
      } else {
        updateData = buildAgentDataFromForm(values, agent);
      }
      
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
      <div className="p-4">
        <div className="flex justify-center items-center h-64">
          <Spin size="large" />
        </div>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="p-4">
        <div className="text-center">Agent not found</div>
        <TremorButton onClick={onClose} className="mt-4">
          Back to Agents List
        </TremorButton>
      </div>
    );
  }

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="p-4">
      <div>
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Agents
        </TremorButton>
        <Title>{agent.agent_name || "Unnamed Agent"}</Title>
        <Text className="text-gray-500 font-mono">{agent.agent_id}</Text>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">Overview</Tab>
          {isAdmin ? <Tab key="settings">Settings</Tab> : <></>}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="Agent ID">{agent.agent_id}</Descriptions.Item>
              <Descriptions.Item label="Agent Name">{agent.agent_name}</Descriptions.Item>
              <Descriptions.Item label="Display Name">{agent.agent_card_params?.name || "-"}</Descriptions.Item>
              <Descriptions.Item label="Description">{agent.agent_card_params?.description || "-"}</Descriptions.Item>
              <Descriptions.Item label="URL">{agent.agent_card_params?.url || "-"}</Descriptions.Item>
              <Descriptions.Item label="Version">{agent.agent_card_params?.version || "-"}</Descriptions.Item>
              <Descriptions.Item label="Protocol Version">{agent.agent_card_params?.protocolVersion || "-"}</Descriptions.Item>
              <Descriptions.Item label="Streaming">
                {agent.agent_card_params?.capabilities?.streaming ? "Yes" : "No"}
              </Descriptions.Item>
              {agent.agent_card_params?.capabilities?.pushNotifications && (
                <Descriptions.Item label="Push Notifications">Yes</Descriptions.Item>
              )}
              {agent.agent_card_params?.capabilities?.stateTransitionHistory && (
                <Descriptions.Item label="State Transition History">Yes</Descriptions.Item>
              )}
              <Descriptions.Item label="Skills">
                {agent.agent_card_params?.skills?.length || 0} configured
              </Descriptions.Item>
              {agent.litellm_params?.model && (
                <Descriptions.Item label="Model">{agent.litellm_params.model}</Descriptions.Item>
              )}
              {agent.litellm_params?.make_public !== undefined && (
                <Descriptions.Item label="Make Public">{agent.litellm_params.make_public ? "Yes" : "No"}</Descriptions.Item>
              )}
              {agent.agent_card_params?.iconUrl && (
                <Descriptions.Item label="Icon URL">{agent.agent_card_params.iconUrl}</Descriptions.Item>
              )}
              {agent.agent_card_params?.documentationUrl && (
                <Descriptions.Item label="Documentation URL">{agent.agent_card_params.documentationUrl}</Descriptions.Item>
              )}
              <Descriptions.Item label="Created At">{formatDate(agent.created_at)}</Descriptions.Item>
              <Descriptions.Item label="Updated At">{formatDate(agent.updated_at)}</Descriptions.Item>
            </Descriptions>

            <AgentCostView agent={agent} />

            {agent.agent_card_params?.skills && agent.agent_card_params.skills.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <Title>Skills</Title>
                <Descriptions bordered column={1} style={{ marginTop: 16 }}>
                  {agent.agent_card_params.skills.map((skill: any, index: number) => (
                    <Descriptions.Item label={skill.name || `Skill ${index + 1}`} key={index}>
                      <div>
                        <div><strong>ID:</strong> {skill.id}</div>
                        <div><strong>Description:</strong> {skill.description}</div>
                        <div><strong>Tags:</strong> {Array.isArray(skill.tags) ? skill.tags.join(", ") : skill.tags}</div>
                        {skill.examples && skill.examples.length > 0 && (
                          <div><strong>Examples:</strong> {Array.isArray(skill.examples) ? skill.examples.join(", ") : skill.examples}</div>
                        )}
                      </div>
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </div>
            )}
          </TabPanel>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabPanel>
              <Card>
                <div className="flex justify-between items-center mb-4">
                  <Title>Agent Settings</Title>
                  {!isEditing && (
                    <TremorButton onClick={() => setIsEditing(true)}>Edit Settings</TremorButton>
                  )}
                </div>

                {isEditing ? (
                  <Form
                    form={form}
                    layout="vertical"
                    onFinish={handleUpdate}
                  >
                    <Form.Item label="Agent ID">
                      <Input value={agent.agent_id} disabled />
                    </Form.Item>

                    {detectedAgentType === "a2a" ? (
                      <AgentFormFields showAgentName={true} />
                    ) : selectedAgentTypeInfo ? (
                      <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} />
                    ) : (
                    <AgentFormFields showAgentName={true} />
                    )}

                    <div className="flex justify-end gap-2 mt-6">
                      <AntButton onClick={() => {
                        setIsEditing(false);
                        fetchAgentInfo();
                      }}>
                        Cancel
                      </AntButton>
                      <TremorButton loading={isSaving}>
                        Save Changes
                      </TremorButton>
                    </div>
                  </Form>
                ) : (
                  <Text>Click &quot;Edit Settings&quot; to modify agent configuration.</Text>
                )}
              </Card>
            </TabPanel>
          )}
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default AgentInfoView;

