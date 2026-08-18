import React, { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Form, Input, InputNumber, Button as AntButton, Spin, Descriptions, Divider } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { ArrowLeft } from "lucide-react";
import { getAgentInfo, patchAgentCall, getAgentCreateMetadata, AgentCreateInfo } from "@/components/networking";
import { Agent } from "@/components/agents/types";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import KeyInfoView from "@/components/templates/key_info_view";
import AgentVirtualKeys from "./agent_virtual_keys";
import AgentFormFields from "./agent_form_fields";
import DynamicAgentFormFields, { buildDynamicAgentData } from "./dynamic_agent_form_fields";
import { buildAgentDataFromForm, parseAgentForForm } from "./agent_config";
import AgentCostView from "./agent_cost_view";
import { detectAgentType, parseDynamicAgentForForm } from "./agent_type_utils";
import AgentCardDiscovery, { DiscoveredAgentCardSelection } from "./agent_card_discovery";
import { buildDiscoveryRequest, overlayDiscoveredCardParams } from "./agent_discovery_utils";

interface AgentInfoViewProps {
  agentId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const AgentInfoView: React.FC<AgentInfoViewProps> = ({ agentId, onClose, accessToken, isAdmin }) => {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const { data: keysData, isLoading: keysLoading, refetch: refetchAgentKeys } = useKeys(1, 100, { agentID: agentId });
  const agentKeys = keysData?.keys ?? [];
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [form] = Form.useForm();
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);
  const [detectedAgentType, setDetectedAgentType] = useState<string>("a2a");
  const [appliedDiscoveredSelection, setAppliedDiscoveredSelection] = useState<DiscoveredAgentCardSelection | null>(
    null,
  );

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
        const typeInfo = agentTypeMetadata.find((t) => t.agent_type === agentType);
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
  };

  // Re-parse form when metadata is loaded
  useEffect(() => {
    if (agent && agentTypeMetadata.length > 0) {
      const agentType = detectAgentType(agent);
      if (agentType !== "a2a") {
        const typeInfo = agentTypeMetadata.find((t) => t.agent_type === agentType);
        if (typeInfo) {
          form.setFieldsValue(parseDynamicAgentForForm(agent, typeInfo));
        }
      }
    }
  }, [agentTypeMetadata, agent]);

  const selectedAgentTypeInfo = agentTypeMetadata.find((t) => t.agent_type === detectedAgentType);
  const watchedFormValues = Form.useWatch([], form);

  const discoveryRequest = useMemo(
    () => buildDiscoveryRequest(detectedAgentType, watchedFormValues || {}, selectedAgentTypeInfo),
    [watchedFormValues, selectedAgentTypeInfo, detectedAgentType],
  );

  const handleApplyDiscoveredCard = (selection: DiscoveredAgentCardSelection | null) => {
    setAppliedDiscoveredSelection(selection);
    if (!selection) return;
    const { selected_card } = selection;
    const skills = (selected_card.skills ?? []).map((s) => ({
      id: s.id ?? "",
      name: s.name ?? "",
      description: s.description ?? "",
      tags: s.tags ?? [],
      examples: s.examples ?? [],
    }));

    const fieldsToSet: Record<string, any> = {
      name: selected_card.name,
      description: selected_card.description,
      url: selection.upstream_url,
      streaming: Boolean(selected_card.capabilities?.streaming),
      skills,
      iconUrl: selected_card.iconUrl,
      documentationUrl: selected_card.documentationUrl,
    };

    const urlCredentialKeys = (selectedAgentTypeInfo?.credential_fields ?? [])
      .map((f) => f.key)
      .filter((key) => /(^|_)(url|api_base|endpoint)$/i.test(key));
    for (const key of urlCredentialKeys) {
      fieldsToSet[key] = selection.upstream_url;
    }

    form.setFieldsValue(fieldsToSet);
  };

  const handleUpdate = async (values: any) => {
    if (!accessToken || !agent) return;

    setIsSaving(true);
    try {
      let updateData: any;

      if (detectedAgentType === "a2a") {
        updateData = buildAgentDataFromForm(values, agent);
      } else if (selectedAgentTypeInfo) {
        updateData = buildDynamicAgentData(values, selectedAgentTypeInfo);
        updateData.agent_name = values.agent_name;
      } else {
        updateData = buildAgentDataFromForm(values, agent);
      }

      if (appliedDiscoveredSelection) {
        updateData = overlayDiscoveredCardParams(updateData, appliedDiscoveredSelection.selected_card);
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
          <Spin size="large" />
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

  if (selectedKey) {
    return (
      <KeyInfoView
        keyId={selectedKey.token}
        keyData={selectedKey}
        onClose={() => setSelectedKey(null)}
        onDelete={() => {
          setSelectedKey(null);
          refetchAgentKeys();
        }}
        teams={null}
        backButtonText="Back to Agent"
      />
    );
  }

  return (
    <div className="p-4">
      <div>
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="size-4" />
          Back to Agents
        </Button>
        <h1 className="text-2xl font-semibold">{agent.agent_name || "Unnamed Agent"}</h1>
        <p className="text-sm text-gray-500 font-mono">{agent.agent_id}</p>
      </div>

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          {isAdmin && (
            <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
              Settings
            </TabsTrigger>
          )}
        </TabsList>

        <div>
          {/* Overview Panel */}
          <TabsContent value="overview" keepMounted>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="Agent ID">{agent.agent_id}</Descriptions.Item>
              <Descriptions.Item label="Agent Name">{agent.agent_name}</Descriptions.Item>
              <Descriptions.Item label="Display Name">{agent.agent_card_params?.name || "-"}</Descriptions.Item>
              <Descriptions.Item label="Description">{agent.agent_card_params?.description || "-"}</Descriptions.Item>
              <Descriptions.Item label="URL">{agent.agent_card_params?.url || "-"}</Descriptions.Item>
              <Descriptions.Item label="Version">{agent.agent_card_params?.version || "-"}</Descriptions.Item>
              <Descriptions.Item label="Protocol Version">
                {agent.agent_card_params?.protocolVersion || "-"}
              </Descriptions.Item>
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
                <Descriptions.Item label="Make Public">
                  {agent.litellm_params.make_public ? "Yes" : "No"}
                </Descriptions.Item>
              )}
              {agent.agent_card_params?.iconUrl && (
                <Descriptions.Item label="Icon URL">{agent.agent_card_params.iconUrl}</Descriptions.Item>
              )}
              {agent.agent_card_params?.documentationUrl && (
                <Descriptions.Item label="Documentation URL">
                  {agent.agent_card_params.documentationUrl}
                </Descriptions.Item>
              )}
              <Descriptions.Item label="TPM Limit">{agent.tpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="RPM Limit">{agent.rpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Session TPM Limit">{agent.session_tpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Session RPM Limit">{agent.session_rpm_limit ?? "Unlimited"}</Descriptions.Item>
              <Descriptions.Item label="Created At">{formatDate(agent.created_at)}</Descriptions.Item>
              <Descriptions.Item label="Updated At">{formatDate(agent.updated_at)}</Descriptions.Item>
            </Descriptions>

            <AgentVirtualKeys keys={agentKeys} isLoading={keysLoading} onKeyClick={setSelectedKey} />

            {agent.object_permission &&
              (agent.object_permission.mcp_servers?.length ||
                agent.object_permission.mcp_access_groups?.length ||
                (agent.object_permission.mcp_tool_permissions &&
                  Object.keys(agent.object_permission.mcp_tool_permissions).length > 0)) && (
                <div style={{ marginTop: 24 }}>
                  <h3 className="text-lg font-medium">MCP Tool Permissions</h3>
                  <Descriptions bordered column={1} style={{ marginTop: 16 }}>
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
                            {Object.entries(agent.object_permission.mcp_tool_permissions).map(([serverId, tools]) => (
                              <div key={serverId}>
                                <span className="font-medium">{serverId}:</span>{" "}
                                {Array.isArray(tools) ? tools.join(", ") : String(tools)}
                              </div>
                            ))}
                          </div>
                        </Descriptions.Item>
                      )}
                  </Descriptions>
                </div>
              )}

            <AgentCostView agent={agent} />

            {agent.agent_card_params?.skills && agent.agent_card_params.skills.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h3 className="text-lg font-medium">Skills</h3>
                <Descriptions bordered column={1} style={{ marginTop: 16 }}>
                  {agent.agent_card_params.skills.map((skill: any, index: number) => (
                    <Descriptions.Item label={skill.name || `Skill ${index + 1}`} key={index}>
                      <div>
                        <div>
                          <strong>ID:</strong> {skill.id}
                        </div>
                        <div>
                          <strong>Description:</strong> {skill.description}
                        </div>
                        <div>
                          <strong>Tags:</strong> {Array.isArray(skill.tags) ? skill.tags.join(", ") : skill.tags}
                        </div>
                        {skill.examples && skill.examples.length > 0 && (
                          <div>
                            <strong>Examples:</strong>{" "}
                            {Array.isArray(skill.examples) ? skill.examples.join(", ") : skill.examples}
                          </div>
                        )}
                      </div>
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </div>
            )}
          </TabsContent>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabsContent value="settings" keepMounted>
              <Card className="block p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Agent Settings</h3>
                  {!isEditing && (
                    <Button
                      onClick={() => {
                        setAppliedDiscoveredSelection(null);
                        setIsEditing(true);
                      }}
                    >
                      Edit Settings
                    </Button>
                  )}
                </div>

                {isEditing ? (
                  <Form form={form} layout="vertical" onFinish={handleUpdate}>
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

                    {discoveryRequest && (
                      <div className="mt-4">
                        <AgentCardDiscovery
                          accessToken={accessToken}
                          onApply={handleApplyDiscoveredCard}
                          discoveryRequest={discoveryRequest}
                          savedAgentCard={agent.agent_card_params ?? null}
                        />
                      </div>
                    )}

                    <Divider />
                    <h3 className="text-lg font-medium mb-4">Rate Limits</h3>
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
                      <AntButton
                        onClick={() => {
                          setAppliedDiscoveredSelection(null);
                          setIsEditing(false);
                          fetchAgentInfo();
                        }}
                      >
                        Cancel
                      </AntButton>
                      <Button type="submit" disabled={isSaving} aria-busy={isSaving}>
                        {isSaving && <UiLoadingSpinner className="size-4" />}
                        Save Changes
                      </Button>
                    </div>
                  </Form>
                ) : (
                  <p>Click &quot;Edit Settings&quot; to modify agent configuration.</p>
                )}
              </Card>
            </TabsContent>
          )}
        </div>
      </Tabs>
    </div>
  );
};

export default AgentInfoView;
