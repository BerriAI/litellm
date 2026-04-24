import React, { useState, useEffect, useCallback } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

function DescRow({
  label,
  children,
}: {
  label: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[220px_1fr] border-t border-border first:border-t-0">
      <dt className="bg-muted px-4 py-3 text-sm font-medium text-foreground border-r border-border">
        {label}
      </dt>
      <dd className="px-4 py-3 text-sm text-foreground">{children}</dd>
    </div>
  );
}

function DescList({ children }: { children: React.ReactNode }) {
  return (
    <dl className="border border-border rounded-md overflow-hidden">
      {children}
    </dl>
  );
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
  const form = useForm<any>({ defaultValues: {} });
  const { register, reset, handleSubmit } = form;
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
        reset(parseAgentForForm(data));
      } else {
        const typeInfo = agentTypeMetadata.find(
          (t) => t.agent_type === agentType,
        );
        if (typeInfo) {
          reset(parseDynamicAgentForForm(data, typeInfo));
        } else {
          reset(parseAgentForForm(data));
        }
      }
    } catch (error) {
      console.error("Error fetching agent info:", error);
      MessageManager.error("Failed to load agent information");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, agentId, agentTypeMetadata, reset]);

  useEffect(() => {
    fetchAgentInfo();
  }, [fetchAgentInfo]);

  useEffect(() => {
    if (agent && agentTypeMetadata.length > 0) {
      const agentType = detectAgentType(agent);
      if (agentType !== "a2a") {
        const typeInfo = agentTypeMetadata.find(
          (t) => t.agent_type === agentType,
        );
        if (typeInfo) {
          reset(parseDynamicAgentForForm(agent, typeInfo));
        }
      }
    }
  }, [agentTypeMetadata, agent, reset]);

  const selectedAgentTypeInfo = agentTypeMetadata.find(
    (t) => t.agent_type === detectedAgentType,
  );

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

      <Tabs defaultValue="overview" className="mt-4">
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {isAdmin && <TabsTrigger value="settings">Settings</TabsTrigger>}
        </TabsList>

        <TabsContent value="overview">
          <DescList>
            <DescRow label="Agent ID">{agent.agent_id}</DescRow>
            <DescRow label="Agent Name">{agent.agent_name}</DescRow>
            <DescRow label="Display Name">
              {agent.agent_card_params?.name || "-"}
            </DescRow>
            <DescRow label="Description">
              {agent.agent_card_params?.description || "-"}
            </DescRow>
            <DescRow label="URL">
              {agent.agent_card_params?.url || "-"}
            </DescRow>
            <DescRow label="Version">
              {agent.agent_card_params?.version || "-"}
            </DescRow>
            <DescRow label="Protocol Version">
              {agent.agent_card_params?.protocolVersion || "-"}
            </DescRow>
            <DescRow label="Streaming">
              {agent.agent_card_params?.capabilities?.streaming ? "Yes" : "No"}
            </DescRow>
            {agent.agent_card_params?.capabilities?.pushNotifications && (
              <DescRow label="Push Notifications">Yes</DescRow>
            )}
            {agent.agent_card_params?.capabilities?.stateTransitionHistory && (
              <DescRow label="State Transition History">Yes</DescRow>
            )}
            <DescRow label="Skills">
              {agent.agent_card_params?.skills?.length || 0} configured
            </DescRow>
            {agent.litellm_params?.model && (
              <DescRow label="Model">{agent.litellm_params.model}</DescRow>
            )}
            {agent.litellm_params?.make_public !== undefined && (
              <DescRow label="Make Public">
                {agent.litellm_params.make_public ? "Yes" : "No"}
              </DescRow>
            )}
            {agent.agent_card_params?.iconUrl && (
              <DescRow label="Icon URL">
                {agent.agent_card_params.iconUrl}
              </DescRow>
            )}
            {agent.agent_card_params?.documentationUrl && (
              <DescRow label="Documentation URL">
                {agent.agent_card_params.documentationUrl}
              </DescRow>
            )}
            <DescRow label="TPM Limit">
              {agent.tpm_limit ?? "Unlimited"}
            </DescRow>
            <DescRow label="RPM Limit">
              {agent.rpm_limit ?? "Unlimited"}
            </DescRow>
            <DescRow label="Session TPM Limit">
              {agent.session_tpm_limit ?? "Unlimited"}
            </DescRow>
            <DescRow label="Session RPM Limit">
              {agent.session_rpm_limit ?? "Unlimited"}
            </DescRow>
            <DescRow label="Created At">{formatDate(agent.created_at)}</DescRow>
            <DescRow label="Updated At">{formatDate(agent.updated_at)}</DescRow>
          </DescList>

          {agent.object_permission &&
            (agent.object_permission.mcp_servers?.length ||
              agent.object_permission.mcp_access_groups?.length ||
              (agent.object_permission.mcp_tool_permissions &&
                Object.keys(agent.object_permission.mcp_tool_permissions)
                  .length > 0)) && (
              <div className="mt-6">
                <h2 className="text-lg font-semibold">
                  MCP Tool Permissions
                </h2>
                <div className="mt-4">
                  <DescList>
                    {agent.object_permission.mcp_servers &&
                      agent.object_permission.mcp_servers.length > 0 && (
                        <DescRow label="MCP Servers">
                          {agent.object_permission.mcp_servers.join(", ")}
                        </DescRow>
                      )}
                    {agent.object_permission.mcp_access_groups &&
                      agent.object_permission.mcp_access_groups.length > 0 && (
                        <DescRow label="MCP Access Groups">
                          {agent.object_permission.mcp_access_groups.join(", ")}
                        </DescRow>
                      )}
                    {agent.object_permission.mcp_tool_permissions &&
                      Object.keys(agent.object_permission.mcp_tool_permissions)
                        .length > 0 && (
                        <DescRow label="Tool permissions per server">
                          <div className="space-y-1">
                            {Object.entries(
                              agent.object_permission.mcp_tool_permissions,
                            ).map(([serverId, tools]) => (
                              <div key={serverId}>
                                <span className="font-medium">{serverId}:</span>{" "}
                                {Array.isArray(tools)
                                  ? tools.join(", ")
                                  : String(tools)}
                              </div>
                            ))}
                          </div>
                        </DescRow>
                      )}
                  </DescList>
                </div>
              </div>
            )}

          <AgentCostView agent={agent} />

          {agent.agent_card_params?.skills &&
            agent.agent_card_params.skills.length > 0 && (
              <div className="mt-6">
                <h2 className="text-lg font-semibold">Skills</h2>
                <div className="mt-4">
                  <DescList>
                    {agent.agent_card_params.skills.map(
                      (skill: any, index: number) => (
                        <DescRow
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
                        </DescRow>
                      ),
                    )}
                  </DescList>
                </div>
              </div>
            )}
        </TabsContent>

        {isAdmin && (
          <TabsContent value="settings">
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
                <FormProvider {...form}>
                  <form
                    onSubmit={handleSubmit(handleUpdate)}
                    className="space-y-4"
                  >
                    <div className="space-y-2">
                      <Label htmlFor="agent-id-readonly">Agent ID</Label>
                      <Input
                        id="agent-id-readonly"
                        value={agent.agent_id}
                        disabled
                        readOnly
                      />
                    </div>

                    {detectedAgentType === "a2a" ? (
                      <AgentFormFields showAgentName={true} />
                    ) : selectedAgentTypeInfo ? (
                      <DynamicAgentFormFields
                        agentTypeInfo={selectedAgentTypeInfo}
                      />
                    ) : (
                      <AgentFormFields showAgentName={true} />
                    )}

                    <hr className="my-4 border-border" />
                    <h3 className="text-lg font-semibold mb-4">Rate Limits</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="tpm_limit">TPM Limit</Label>
                        <Input
                          id="tpm_limit"
                          type="number"
                          min={0}
                          placeholder="Unlimited"
                          {...register("tpm_limit", { valueAsNumber: true })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="rpm_limit">RPM Limit</Label>
                        <Input
                          id="rpm_limit"
                          type="number"
                          min={0}
                          placeholder="Unlimited"
                          {...register("rpm_limit", { valueAsNumber: true })}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="session_tpm_limit">
                          Session TPM Limit
                        </Label>
                        <Input
                          id="session_tpm_limit"
                          type="number"
                          min={0}
                          placeholder="Unlimited"
                          {...register("session_tpm_limit", {
                            valueAsNumber: true,
                          })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="session_rpm_limit">
                          Session RPM Limit
                        </Label>
                        <Input
                          id="session_rpm_limit"
                          type="number"
                          min={0}
                          placeholder="Unlimited"
                          {...register("session_rpm_limit", {
                            valueAsNumber: true,
                          })}
                        />
                      </div>
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
                  </form>
                </FormProvider>
              ) : (
                <p className="text-muted-foreground">
                  Click &quot;Edit Settings&quot; to modify agent configuration.
                </p>
              )}
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
};

export default AgentInfoView;
