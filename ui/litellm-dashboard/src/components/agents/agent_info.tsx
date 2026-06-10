import React, { useState, useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Card, Title, Text, Button as TremorButton, Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import { Form, Input, InputNumber, Button as AntButton, Spin, Descriptions, Divider } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { getAgentInfo, patchAgentCall, getAgentCreateMetadata, AgentCreateInfo } from "../networking";
import { Agent } from "./types";
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
  const { t } = useTranslation();
  const [agent, setAgent] = useState<Agent | null>(null);
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
      MessageManager.error(t("agentsPage.agentInfo.failedToLoad"));
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
      MessageManager.success(t("agentsPage.agentInfo.updateSuccess"));
      setIsEditing(false);
      fetchAgentInfo();
    } catch (error) {
      console.error("Error updating agent:", error);
      MessageManager.error(t("agentsPage.agentInfo.updateFailed"));
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
        <div className="text-center">{t("agentsPage.agentInfo.agentNotFound")}</div>
        <TremorButton onClick={onClose} className="mt-4">
          {t("agentsPage.agentInfo.backToList")}
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
          {t("agentsPage.agentInfo.backToAgents")}
        </TremorButton>
        <Title>{agent.agent_name || t("agentsPage.agentInfo.unnamedAgent")}</Title>
        <Text className="text-gray-500 font-mono">{agent.agent_id}</Text>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">{t("agentsPage.agentInfo.overviewTab")}</Tab>
          {isAdmin ? <Tab key="settings">{t("agentsPage.agentInfo.settingsTab")}</Tab> : <></>}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Descriptions bordered column={1}>
              <Descriptions.Item label={t("agentsPage.agentInfo.agentIdLabel")}>{agent.agent_id}</Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.agentNameLabel")}>{agent.agent_name}</Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.displayNameLabel")}>
                {agent.agent_card_params?.name || "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("common.description")}>
                {agent.agent_card_params?.description || "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.urlLabel")}>
                {agent.agent_card_params?.url || "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.versionLabel")}>
                {agent.agent_card_params?.version || "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.protocolVersionLabel")}>
                {agent.agent_card_params?.protocolVersion || "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.streamingLabel")}>
                {agent.agent_card_params?.capabilities?.streaming ? t("common.yes") : t("common.no")}
              </Descriptions.Item>
              {agent.agent_card_params?.capabilities?.pushNotifications && (
                <Descriptions.Item label={t("agentsPage.agentInfo.pushNotificationsLabel")}>
                  {t("common.yes")}
                </Descriptions.Item>
              )}
              {agent.agent_card_params?.capabilities?.stateTransitionHistory && (
                <Descriptions.Item label={t("agentsPage.agentInfo.stateTransitionHistoryLabel")}>
                  {t("common.yes")}
                </Descriptions.Item>
              )}
              <Descriptions.Item label={t("agentsPage.agentInfo.skillsLabel")}>
                {t("agentsPage.agentInfo.skillsConfigured", {
                  count: agent.agent_card_params?.skills?.length || 0,
                })}
              </Descriptions.Item>
              {agent.litellm_params?.model && (
                <Descriptions.Item label={t("agentsPage.agentInfo.modelLabel")}>
                  {agent.litellm_params.model}
                </Descriptions.Item>
              )}
              {agent.litellm_params?.make_public !== undefined && (
                <Descriptions.Item label={t("agentsPage.agentInfo.makePublicLabel")}>
                  {agent.litellm_params.make_public ? t("common.yes") : t("common.no")}
                </Descriptions.Item>
              )}
              {agent.agent_card_params?.iconUrl && (
                <Descriptions.Item label={t("agentsPage.agentInfo.iconUrlLabel")}>
                  {agent.agent_card_params.iconUrl}
                </Descriptions.Item>
              )}
              {agent.agent_card_params?.documentationUrl && (
                <Descriptions.Item label={t("agentsPage.agentInfo.documentationUrlLabel")}>
                  {agent.agent_card_params.documentationUrl}
                </Descriptions.Item>
              )}
              <Descriptions.Item label={t("agentsPage.agentInfo.tpmLimitLabel")}>
                {agent.tpm_limit ?? t("agentsPage.agentInfo.unlimited")}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.rpmLimitLabel")}>
                {agent.rpm_limit ?? t("agentsPage.agentInfo.unlimited")}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.sessionTpmLimitLabel")}>
                {agent.session_tpm_limit ?? t("agentsPage.agentInfo.unlimited")}
              </Descriptions.Item>
              <Descriptions.Item label={t("agentsPage.agentInfo.sessionRpmLimitLabel")}>
                {agent.session_rpm_limit ?? t("agentsPage.agentInfo.unlimited")}
              </Descriptions.Item>
              <Descriptions.Item label={t("common.createdAt")}>{formatDate(agent.created_at)}</Descriptions.Item>
              <Descriptions.Item label={t("common.updatedAt")}>{formatDate(agent.updated_at)}</Descriptions.Item>
            </Descriptions>

            {agent.object_permission &&
              (agent.object_permission.mcp_servers?.length ||
                agent.object_permission.mcp_access_groups?.length ||
                (agent.object_permission.mcp_tool_permissions &&
                  Object.keys(agent.object_permission.mcp_tool_permissions).length > 0)) && (
                <div style={{ marginTop: 24 }}>
                  <Title>{t("agentsPage.agentInfo.mcpToolPermissionsTitle")}</Title>
                  <Descriptions bordered column={1} style={{ marginTop: 16 }}>
                    {agent.object_permission.mcp_servers && agent.object_permission.mcp_servers.length > 0 && (
                      <Descriptions.Item label={t("agentsPage.agentInfo.mcpServersLabel")}>
                        {agent.object_permission.mcp_servers.join(", ")}
                      </Descriptions.Item>
                    )}
                    {agent.object_permission.mcp_access_groups &&
                      agent.object_permission.mcp_access_groups.length > 0 && (
                        <Descriptions.Item label={t("agentsPage.agentInfo.mcpAccessGroupsLabel")}>
                          {agent.object_permission.mcp_access_groups.join(", ")}
                        </Descriptions.Item>
                      )}
                    {agent.object_permission.mcp_tool_permissions &&
                      Object.keys(agent.object_permission.mcp_tool_permissions).length > 0 && (
                        <Descriptions.Item label={t("agentsPage.agentInfo.toolPermissionsPerServerLabel")}>
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
                <Title>{t("agentsPage.agentInfo.skillsSectionTitle")}</Title>
                <Descriptions bordered column={1} style={{ marginTop: 16 }}>
                  {agent.agent_card_params.skills.map((skill: any, index: number) => (
                    <Descriptions.Item
                      label={skill.name || t("agentsPage.agentInfo.skillFallbackLabel", { index: index + 1 })}
                      key={index}
                    >
                      <div>
                        <div>
                          <strong>{t("agentsPage.agentInfo.skillIdField")}:</strong> {skill.id}
                        </div>
                        <div>
                          <strong>{t("common.description")}:</strong> {skill.description}
                        </div>
                        <div>
                          <strong>{t("agentsPage.agentInfo.skillTagsField")}:</strong>{" "}
                          {Array.isArray(skill.tags) ? skill.tags.join(", ") : skill.tags}
                        </div>
                        {skill.examples && skill.examples.length > 0 && (
                          <div>
                            <strong>{t("agentsPage.agentInfo.skillExamplesField")}:</strong>{" "}
                            {Array.isArray(skill.examples) ? skill.examples.join(", ") : skill.examples}
                          </div>
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
                  <Title>{t("agentsPage.agentInfo.agentSettingsTitle")}</Title>
                  {!isEditing && (
                    <TremorButton
                      onClick={() => {
                        setAppliedDiscoveredSelection(null);
                        setIsEditing(true);
                      }}
                    >
                      {t("agentsPage.agentInfo.editSettings")}
                    </TremorButton>
                  )}
                </div>

                {isEditing ? (
                  <Form form={form} layout="vertical" onFinish={handleUpdate}>
                    <Form.Item label={t("agentsPage.agentInfo.agentIdLabel")}>
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
                    <Title className="mb-4">{t("agentsPage.agentInfo.rateLimitsTitle")}</Title>
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item label={t("agentsPage.agentInfo.tpmLimitLabel")} name="tpm_limit">
                        <InputNumber className="w-full" min={0} placeholder={t("agentsPage.agentInfo.unlimited")} />
                      </Form.Item>
                      <Form.Item label={t("agentsPage.agentInfo.rpmLimitLabel")} name="rpm_limit">
                        <InputNumber className="w-full" min={0} placeholder={t("agentsPage.agentInfo.unlimited")} />
                      </Form.Item>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item label={t("agentsPage.agentInfo.sessionTpmLimitLabel")} name="session_tpm_limit">
                        <InputNumber className="w-full" min={0} placeholder={t("agentsPage.agentInfo.unlimited")} />
                      </Form.Item>
                      <Form.Item label={t("agentsPage.agentInfo.sessionRpmLimitLabel")} name="session_rpm_limit">
                        <InputNumber className="w-full" min={0} placeholder={t("agentsPage.agentInfo.unlimited")} />
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
                        {t("common.cancel")}
                      </AntButton>
                      <TremorButton loading={isSaving}>{t("agentsPage.agentInfo.saveChanges")}</TremorButton>
                    </div>
                  </Form>
                ) : (
                  <Text>{t("agentsPage.agentInfo.editSettingsHint")}</Text>
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
