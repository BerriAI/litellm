import React, { useState, useEffect, useCallback } from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import { Form, Input, InputNumber, Descriptions } from "antd";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Loader2 } from "lucide-react";
import MessageManager from "@/components/molecules/message_manager";
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

  const fetchAgentInfo = useCallback(async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const data = await getAgentInfo(accessToken, agentId);
      setAgent(data);

      const agentType = detectAgentType(data);
      setDetectedAgentType(agentType);

      if (agentType === "a2a") {
        form.setFieldsValue(parseAgentForForm(data));
      } else {
        const typeInfo = agentTypeMetadata.find(
          (t) => t.agent_type === agentType,
        );
        if (typeInfo) {
          form.setFieldsValue(parseDynamicAgentForForm(data, typeInfo));
        } else {
          form.setFieldsValue(parseAgentForForm(data));
        }
      }
    } catch (error) {
      console.error("Error fetching agent info:", error);
      MessageManager.error("Failed to load agent information");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, agentId, agentTypeMetadata, form]);

  useEffect(() => {
    fetchAgentInfo();
  }, [fetchAgentInfo]);

  // Re-parse form when metadata is loaded
  useEffect(() => {
    if (agent && agentTypeMetadata.length > 0) {
      const agentType = detectAgentType(agent);
      if (agentType !== "a2a") {
        const typeInfo = agentTypeMetadata.find(
          (t) => t.agent_type === agentType,
        );
        if (typeInfo) {
          form.setFieldsValue(parseDynamicAgentForForm(agent, typeInfo));
        }
      }
    }
  }, [agentTypeMetadata, agent, form]);

  const selectedAgentTypeInfo = agentTypeMetadata.find(
    (t) => t.agent_type === detectedAgentType,
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleUpdate = async (values: any) => {
    if (!accessToken || !agent) return;

    setIsSaving(true);
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
      MessageManager.success("Agent updated successfully");
      setIsEditing(false);
      fetchAgentInfo();
    } catch (error) {
      console.error("Error updating agent:", error);
      MessageManager.error("Failed to update agent");
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <div className="flex justify-center items-center h-64">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="p-4">
        <div className="text-center">Agent not found</div>
        <Button onClick={onClose} className="mt-4">
          Back to Agents List
        </Button>
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
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Agents
        </Button>
        <h1 className="text-2xl font-semibold">
          {agent.agent_name || "Unnamed Agent"}
        </h1>
        <span className="text-muted-foreground font-mono">{agent.agent_id}</span>
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
              <Descriptions.Item label="TPM Limit">{agent.tpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="RPM Limit">{agent.rpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Session TPM Limit">{agent.session_tpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Session RPM Limit">{agent.session_rpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Created At">{formatDate(agent.created_at)}</Descriptions.Item>
              <Descriptions.Item label="Updated At">{formatDate(agent.updated_at)}</Descriptions.Item>
            </Descriptions>

            {agent.object_permission &&
              (agent.object_permission.mcp_servers?.length ||
                agent.object_permission.mcp_access_groups?.length ||
                (agent.object_permission.mcp_tool_permissions &&
                  Object.keys(agent.object_permission.mcp_tool_permissions).length > 0)) && (
              <div className="mt-6">
                <h2 className="text-lg font-semibold">MCP Tool Permissions</h2>
                <Descriptions bordered column={1} className="mt-4">
                  {agent.object_permission.mcp_servers && agent.object_permission.mcp_servers.length > 0 && (
                    <Descriptions.Item label="MCP Servers">
                      {agent.object_permission.mcp_servers.join(", ")}
                    </Descriptions.Item>
                  )}
                  {agent.object_permission.mcp_access_groups &&
                    agent.object_permission.mcp_access_groups.length > 0 && (
                      <Descriptions.Item label="MCP Access Groups">
                        {agent.object_permission.mcp_access_groups.join(", ")}
                      </Descriptions.Item>
                    )}
                  {agent.object_permission.mcp_tool_permissions &&
                    Object.keys(agent.object_permission.mcp_tool_permissions).length > 0 && (
                      <Descriptions.Item label="Tool permissions per server">
                        <div className="space-y-1">
                          {Object.entries(agent.object_permission.mcp_tool_permissions).map(
                            ([serverId, tools]) => (
                              <div key={serverId}>
                                <span className="font-medium">{serverId}:</span>{" "}
                                {Array.isArray(tools) ? tools.join(", ") : String(tools)}
                              </div>
                            )
                          )}
                        </div>
                      </Descriptions.Item>
                    )}
                </Descriptions>
              </div>
            )}

            <AgentCostView agent={agent} />

            {agent.agent_card_params?.skills &&
              agent.agent_card_params.skills.length > 0 && (
                <div className="mt-6">
                  <h2 className="text-lg font-semibold">Skills</h2>
                  <Descriptions bordered column={1} className="mt-4">
                    {agent.agent_card_params.skills.map(
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      (skill: any, index: number) => (
                        <Descriptions.Item
                          label={skill.name || `Skill ${index + 1}`}
                          key={index}
                        >
                          <div>
                            <div>
                              <strong>ID:</strong> {skill.id}
                            </div>
                            <div>
                              <strong>Description:</strong> {skill.description}
                            </div>
                            <div>
                              <strong>Tags:</strong>{" "}
                              {Array.isArray(skill.tags)
                                ? skill.tags.join(", ")
                                : skill.tags}
                            </div>
                            {skill.examples && skill.examples.length > 0 && (
                              <div>
                                <strong>Examples:</strong>{" "}
                                {Array.isArray(skill.examples)
                                  ? skill.examples.join(", ")
                                  : skill.examples}
                              </div>
                            )}
                          </div>
                        </Descriptions.Item>
                      ),
                    )}
                  </Descriptions>
                </div>
              )}
          </TabPanel>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabPanel>
              <Card className="p-4">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">Agent Settings</h2>
                  {!isEditing && (
                    <Button onClick={() => setIsEditing(true)}>
                      Edit Settings
                    </Button>
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

                    <hr className="my-4 border-border" />
                    <h3 className="text-lg font-semibold mb-4">Rate Limits</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item label="TPM Limit" name="tpm_limit">
                        <InputNumber className="w-full" min={0} placeholder="Unlimited" />
                      </Form.Item>
                      <Form.Item label="RPM Limit" name="rpm_limit">
                        <InputNumber className="w-full" min={0} placeholder="Unlimited" />
                      </Form.Item>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item label="Session TPM Limit" name="session_tpm_limit">
                        <InputNumber className="w-full" min={0} placeholder="Unlimited" />
                      </Form.Item>
                      <Form.Item label="Session RPM Limit" name="session_rpm_limit">
                        <InputNumber className="w-full" min={0} placeholder="Unlimited" />
                      </Form.Item>
                    </div>

                    <div className="flex justify-end gap-2 mt-6">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => {
                          setIsEditing(false);
                          fetchAgentInfo();
                        }}
                      >
                        Cancel
                      </Button>
                      <Button type="submit" disabled={isSaving}>
                        {isSaving ? "Saving..." : "Save Changes"}
                      </Button>
                    </div>
                  </Form>
                ) : (
                  <p className="text-muted-foreground">
                    Click &quot;Edit Settings&quot; to modify agent
                    configuration.
                  </p>
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

