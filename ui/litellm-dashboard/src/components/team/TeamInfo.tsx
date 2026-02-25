import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import UserSearchModal from "@/components/common_components/user_search_modal";
import {
  getGuardrailsList,
  getPoliciesList,
  getPolicyInfoWithGuardrails,
  Member,
  Organization,
  organizationInfoCall,
  teamInfoCall,
  teamMemberAddCall,
  teamMemberDeleteCall,
  teamMemberUpdateCall,
  teamUpdateCall,
} from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { mapEmptyStringToNull } from "@/utils/keyUpdateUtils";
import { isProxyAdminRole } from "@/utils/roles";
import { EditOutlined, InfoCircleOutlined, SaveOutlined } from "@ant-design/icons";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Badge, Card, Grid, Text, TextInput, Title } from "@tremor/react";
import { Button, Form, Input, message, Select, Switch, Tabs, Tooltip } from "antd";
import { CheckIcon, CopyIcon } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import AgentSelector from "../agent_management/AgentSelector";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import DurationSelect from "../common_components/DurationSelect";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import { unfurlWildcardModelsInList } from "../key_team_helpers/fetch_available_models_team_key";
import LoggingSettingsView from "../logging_settings_view";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import { ModelSelect } from "../ModelSelect/ModelSelect";
import NotificationsManager from "../molecules/notifications_manager";
import { fetchMCPAccessGroups } from "../networking";
import ObjectPermissionsView from "../object_permissions_view";
import NumericalInput from "../shared/numerical_input";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import EditLoggingSettings from "./EditLoggingSettings";
import MemberModal from "./EditMembership";
import MemberPermissions from "./member_permissions";
import {
  getTeamInfoDefaultTab,
  getTeamInfoVisibleTabs,
  TEAM_INFO_TAB_KEYS,
  TEAM_INFO_TAB_LABELS,
} from "./tabVisibilityUtils";
import TeamMembersComponent from "./TeamMemberTab";

export interface TeamMembership {
  user_id: string;
  team_id: string;
  budget_id: string;
  spend: number;
  litellm_budget_table: {
    budget_id: string;
    soft_budget: number | null;
    max_budget: number | null;
    max_parallel_requests: number | null;
    tpm_limit: number | null;
    rpm_limit: number | null;
    model_max_budget: Record<string, number> | null;
    budget_duration: string | null;
  };
}

export interface TeamData {
  team_id: string;
  team_info: {
    team_alias: string;
    team_id: string;
    organization_id: string | null;
    admins: string[];
    members: string[];
    members_with_roles: Member[];
    metadata: Record<string, any>;
    tpm_limit: number | null;
    rpm_limit: number | null;
    max_budget: number | null;
    soft_budget?: number | null;
    budget_duration: string | null;
    models: string[];
    blocked: boolean;
    spend: number;
    max_parallel_requests: number | null;
    budget_reset_at: string | null;
    model_id: string | null;
    litellm_model_table: {
      model_aliases: Record<string, string>;
    } | null;
    created_at: string;
    access_group_ids?: string[];
    guardrails?: string[];
    policies?: string[];
    object_permission?: {
      object_permission_id: string;
      mcp_servers: string[];
      mcp_access_groups?: string[];
      mcp_tool_permissions?: Record<string, string[]>;
      vector_stores: string[];
      agents?: string[];
      agent_access_groups?: string[];
    };
    team_member_budget_table: {
      max_budget: number;
      budget_duration: string;
      tpm_limit: number | null;
      rpm_limit: number | null;
    } | null;
  };
  keys: any[];
  team_memberships: TeamMembership[];
}

export interface TeamInfoProps {
  teamId: string;
  onUpdate: (data: any) => void;
  onClose: () => void;
  accessToken: string | null;
  is_team_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editTeam: boolean;
  premiumUser?: boolean;
}

const getOrganizationModels = (organization: Organization | null, userModels: string[]) => {
  let tempModelsToPick = [];

  if (organization) {
    // Check if organization has "all-proxy-models" in its models array
    if (organization.models.includes("all-proxy-models")) {
      // Treat as all-proxy-models (use userModels)
      tempModelsToPick = userModels;
    } else if (organization.models.length > 0) {
      // Organization has specific models
      tempModelsToPick = organization.models;
    } else {
      // Empty array [] is treated as all-proxy-models
      tempModelsToPick = userModels;
    }
  } else {
    // No organization, show all available models
    tempModelsToPick = userModels;
  }

  return unfurlWildcardModelsInList(tempModelsToPick, userModels);
};

const TeamInfoView: React.FC<TeamInfoProps> = ({
  teamId,
  onClose,
  accessToken,
  is_team_admin,
  is_proxy_admin,
  userModels,
  editTeam,
  premiumUser = false,
  onUpdate,
}) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [selectedEditMember, setSelectedEditMember] = useState<Member | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [policyGuardrails, setPolicyGuardrails] = useState<Record<string, string[]>>({});
  const [loadingPolicies, setLoadingPolicies] = useState(false);
  const [memberToDelete, setMemberToDelete] = useState<Member | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isTeamSaving, setIsTeamSaving] = useState(false);
  const [organization, setOrganization] = useState<Organization | null>(null);
  const { userRole } = useAuthorized();

  const canEditTeam = is_team_admin || is_proxy_admin;
  const visibleTabs = useMemo(() => getTeamInfoVisibleTabs(canEditTeam), [canEditTeam]);
  const defaultTabKey = useMemo(
    () => getTeamInfoDefaultTab(editTeam, canEditTeam),
    [editTeam, canEditTeam]
  );

  const fetchTeamInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await teamInfoCall(accessToken, teamId);
      setTeamData(response);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to load team information");
      console.error("Error fetching team info:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTeamInfo();
  }, [teamId, accessToken]);

  // Fetch organization data when team has organization_id
  useEffect(() => {
    const fetchOrganization = async () => {
      if (!accessToken || !teamData?.team_info?.organization_id) {
        setOrganization(null);
        return;
      }

      try {
        const orgData = await organizationInfoCall(accessToken, teamData.team_info.organization_id);
        setOrganization(orgData);
      } catch (error) {
        console.error("Error fetching organization info:", error);
        setOrganization(null);
      }
    };

    fetchOrganization();
  }, [accessToken, teamData?.team_info?.organization_id]);

  // Compute modelsToPick based on organization and userModels
  const modelsToPick = useMemo(() => {
    return getOrganizationModels(organization, userModels);
  }, [organization, userModels]);

  const fetchMcpAccessGroups = async () => {
    if (!accessToken) return;
    if (mcpAccessGroupsLoaded) return;
    try {
      const groups = await fetchMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups);
      setMcpAccessGroupsLoaded(true);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (!accessToken) return;
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        if (!accessToken) return;
        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    fetchGuardrails();
    fetchPolicies();
  }, [accessToken]);

  // Fetch resolved guardrails for all policies
  useEffect(() => {
    const fetchPolicyGuardrails = async () => {
      if (!accessToken || !teamData?.team_info?.policies || teamData.team_info.policies.length === 0) {
        return;
      }

      setLoadingPolicies(true);
      const guardrailsMap: Record<string, string[]> = {};

      try {
        await Promise.all(
          teamData.team_info.policies.map(async (policyName: string) => {
            try {
              const policyInfo = await getPolicyInfoWithGuardrails(accessToken, policyName);
              guardrailsMap[policyName] = policyInfo.resolved_guardrails || [];
            } catch (error) {
              console.error(`Failed to fetch guardrails for policy ${policyName}:`, error);
              guardrailsMap[policyName] = [];
            }
          })
        );
        setPolicyGuardrails(guardrailsMap);
      } catch (error) {
        console.error("Failed to fetch policy guardrails:", error);
      } finally {
        setLoadingPolicies(false);
      }
    };

    fetchPolicyGuardrails();
  }, [accessToken, teamData?.team_info?.policies]);

  const handleMemberCreate = async (values: any) => {
    try {
      if (accessToken == null) return;

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
      };

      await teamMemberAddCall(accessToken, teamId, member);

      NotificationsManager.success("Team member added successfully");
      setIsAddMemberModalVisible(false);
      form.resetFields();

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = "Failed to add team member";

      if (error?.raw?.detail?.error?.includes("Assigning team admins is a premium feature")) {
        errMsg = "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
      } else if (error?.message) {
        errMsg = error.message;
      }

      NotificationsManager.fromBackend(errMsg);
      console.error("Error adding team member:", error);
    }
  };

  const handleMemberUpdate = async (values: any) => {
    try {
      if (accessToken == null) {
        return;
      }

      const member: Member = {
        user_email: values.user_email,
        user_id: values.user_id,
        role: values.role,
        max_budget_in_team: values.max_budget_in_team,
        tpm_limit: values.tpm_limit,
        rpm_limit: values.rpm_limit,
      };
      message.destroy(); // Remove all existing toasts

      await teamMemberUpdateCall(accessToken, teamId, member);

      NotificationsManager.success("Team member updated successfully");
      setIsEditMemberModalVisible(false);

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = "Failed to update team member";
      if (error?.raw?.detail?.includes("Assigning team admins is a premium feature")) {
        errMsg = "Assigning admins is an enterprise-only feature. Please upgrade your LiteLLM plan to enable this.";
      } else if (error?.message) {
        errMsg = error.message;
      }
      setIsEditMemberModalVisible(false);

      message.destroy(); // Remove all existing toasts

      NotificationsManager.fromBackend(errMsg);
      console.error("Error updating team member:", error);
    }
  };

  const handleMemberDelete = (member: Member) => {
    setMemberToDelete(member);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!memberToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await teamMemberDeleteCall(accessToken, teamId, memberToDelete);

      NotificationsManager.success("Team member removed successfully");

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to remove team member");
      console.error("Error removing team member:", error);
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setMemberToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
    setMemberToDelete(null);
  };

  const handleTeamUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      setIsTeamSaving(true);

      let parsedMetadata = {};
      try {
        const rawMetadata = values.metadata ? JSON.parse(values.metadata) : {};
        // Exclude soft_budget_alerting_emails from parsed metadata since it's handled separately
        const { soft_budget_alerting_emails, ...rest } = rawMetadata;
        parsedMetadata = rest;
      } catch (e) {
        NotificationsManager.fromBackend("Invalid JSON in metadata field");
        return;
      }

      let secretManagerSettings: Record<string, any> | undefined;
      if (typeof values.secret_manager_settings === "string") {
        const trimmedSecretConfig = values.secret_manager_settings.trim();
        if (trimmedSecretConfig.length > 0) {
          try {
            secretManagerSettings = JSON.parse(values.secret_manager_settings);
          } catch (e) {
            NotificationsManager.fromBackend("Invalid JSON in secret manager settings");
            return;
          }
        }
      }

      const sanitizeNumeric = (v: any) => {
        if (v === null || v === undefined) return null;
        if (typeof v === "string" && v.trim() === "") return null;
        if (typeof v === "number" && Number.isNaN(v)) return null;
        return v;
      };

      const updateData: any = {
        team_id: teamId,
        team_alias: values.team_alias,
        models: values.models,
        tpm_limit: sanitizeNumeric(values.tpm_limit),
        rpm_limit: sanitizeNumeric(values.rpm_limit),
        max_budget: values.max_budget,
        soft_budget: sanitizeNumeric(values.soft_budget),
        budget_duration: values.budget_duration,
        metadata: {
          ...parsedMetadata,
          ...(values.guardrails?.length > 0 ? { guardrails: values.guardrails } : {}),
          ...(values.logging_settings?.length > 0 ? { logging: values.logging_settings } : {}),
          disable_global_guardrails: values.disable_global_guardrails || false,
          soft_budget_alerting_emails:
            typeof values.soft_budget_alerting_emails === "string"
              ? values.soft_budget_alerting_emails
                .split(",")
                .map((email: string) => email.trim())
                .filter((email: string) => email.length > 0)
              : values.soft_budget_alerting_emails || [],
          ...(secretManagerSettings !== undefined ? { secret_manager_settings: secretManagerSettings } : {}),
        },
        ...(values.policies?.length > 0 ? { policies: values.policies } : {}),
        organization_id: values.organization_id,
      };

      updateData.max_budget = mapEmptyStringToNull(updateData.max_budget);
      updateData.team_member_budget_duration = values.team_member_budget_duration;

      if (values.team_member_budget !== undefined) {
        updateData.team_member_budget = Number(values.team_member_budget);
      }

      if (values.team_member_key_duration !== undefined) {
        updateData.team_member_key_duration = values.team_member_key_duration;
      }

      if (values.team_member_tpm_limit !== undefined || values.team_member_rpm_limit !== undefined) {
        updateData.team_member_tpm_limit = sanitizeNumeric(values.team_member_tpm_limit);
        updateData.team_member_rpm_limit = sanitizeNumeric(values.team_member_rpm_limit);
      }

      // Handle object_permission updates
      const { servers, accessGroups } = values.mcp_servers_and_groups || {
        servers: [],
        accessGroups: [],
      };
      const serverIds = new Set(servers || []);
      const mcpToolPermissions = Object.fromEntries(
        Object.entries(values.mcp_tool_permissions || {}).filter(([serverId]) => serverIds.has(serverId)),
      );

      updateData.object_permission = {};
      if (servers) {
        updateData.object_permission.mcp_servers = servers;
      }
      if (accessGroups) {
        updateData.object_permission.mcp_access_groups = accessGroups;
      }
      if (mcpToolPermissions) {
        updateData.object_permission.mcp_tool_permissions = mcpToolPermissions;
      }
      delete values.mcp_servers_and_groups;
      delete values.mcp_tool_permissions;

      // Handle agent permissions
      const { agents, accessGroups: agentAccessGroups } = values.agents_and_groups || {
        agents: [],
        accessGroups: [],
      };
      if (agents && agents.length > 0) {
        updateData.object_permission.agents = agents;
      }
      if (agentAccessGroups && agentAccessGroups.length > 0) {
        updateData.object_permission.agent_access_groups = agentAccessGroups;
      }
      delete values.agents_and_groups;

      // Handle vector stores permissions
      if (values.vector_stores && values.vector_stores.length > 0) {
        updateData.object_permission.vector_stores = values.vector_stores;
      }

      // Pass access_group_ids to the update request
      if (values.access_group_ids !== undefined) {
        updateData.access_group_ids = values.access_group_ids;
      }

      const response = await teamUpdateCall(accessToken, updateData);

      NotificationsManager.success("Team settings updated successfully");
      setIsEditing(false);
      fetchTeamInfo();
    } catch (error) {
      console.error("Error updating team:", error);
    } finally {
      setIsTeamSaving(false);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!teamData?.team_info) {
    return <div className="p-4">Team not found</div>;
  }

  const { team_info: info } = teamData;

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button
            type="text"
            icon={<ArrowLeftIcon className="h-4 w-4" />}
            onClick={onClose}
            className="mb-4"
          >
            Back to Teams
          </Button>
          <Title>{info.team_alias}</Title>
          <div className="flex items-center">
            <Text className="text-gray-500 font-mono">{info.team_id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["team-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(info.team_id, "team-id")}
              className={`left-2 z-10 transition-all duration-200 ${copiedStates["team-id"]
                ? "text-green-600 bg-green-50 border-green-200"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
            />
          </div>
        </div>
      </div>

      <Tabs
        defaultActiveKey={defaultTabKey}
        className="mb-4"
        items={[
          {
            key: TEAM_INFO_TAB_KEYS.OVERVIEW,
            label: TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.OVERVIEW],
            children: (
              <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
                <Card>
                  <Text>Budget Status</Text>
                  <div className="mt-2">
                    <Title>${formatNumberWithCommas(info.spend, 4)}</Title>
                    <Text>
                      of {info.max_budget === null ? "Unlimited" : `$${formatNumberWithCommas(info.max_budget, 4)}`}
                    </Text>
                    {info.budget_duration && <Text className="text-gray-500">Reset: {info.budget_duration}</Text>}
                    <br />
                    {info.team_member_budget_table && (
                      <Text className="text-gray-500">
                        Team Member Budget: ${formatNumberWithCommas(info.team_member_budget_table.max_budget, 4)}
                      </Text>
                    )}
                  </div>
                </Card>

                <Card>
                  <Text>Rate Limits</Text>
                  <div className="mt-2">
                    <Text>TPM: {info.tpm_limit || "Unlimited"}</Text>
                    <Text>RPM: {info.rpm_limit || "Unlimited"}</Text>
                    {info.max_parallel_requests && <Text>Max Parallel Requests: {info.max_parallel_requests}</Text>}
                  </div>
                </Card>

                <Card>
                  <Text>Models</Text>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {info.models.length === 0 ? (
                      <Badge color="red">All proxy models</Badge>
                    ) : (
                      info.models.map((model, index) => (
                        <Badge key={index} color="red">
                          {model}
                        </Badge>
                      ))
                    )}
                  </div>
                </Card>

                <Card>
                  <Text className="font-semibold text-gray-900">Virtual Keys</Text>
                  <div className="mt-2">
                    <Text>User Keys: {teamData.keys.filter((key) => key.user_id).length}</Text>
                    <Text>Service Account Keys: {teamData.keys.filter((key) => !key.user_id).length}</Text>
                    <Text className="text-gray-500">Total: {teamData.keys.length}</Text>
                  </div>
                </Card>

                <ObjectPermissionsView
                  objectPermission={info.object_permission}
                  variant="card"
                  accessToken={accessToken}
                />

                <Card>
                  <Text className="font-semibold text-gray-900 mb-3">Guardrails</Text>
                  {info.guardrails && info.guardrails.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {info.guardrails.map((guardrail: string, index: number) => (
                        <Badge key={index} color="blue">
                          {guardrail}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <Text className="text-gray-500">No guardrails configured</Text>
                  )}
                  {info.metadata?.disable_global_guardrails && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <Badge color="yellow">Global Guardrails Disabled</Badge>
                    </div>
                  )}
                </Card>

                <Card>
                  <Text className="font-semibold text-gray-900 mb-3">Policies</Text>
                  {info.policies && info.policies.length > 0 ? (
                    <div className="space-y-4">
                      {info.policies.map((policy: string, index: number) => (
                        <div key={index} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Badge color="purple">{policy}</Badge>
                            {loadingPolicies && <Text className="text-xs text-gray-400">Loading guardrails...</Text>}
                          </div>
                          {!loadingPolicies && policyGuardrails[policy] && policyGuardrails[policy].length > 0 && (
                            <div className="ml-4 pl-3 border-l-2 border-gray-200">
                              <Text className="text-xs text-gray-500 mb-1">Resolved Guardrails:</Text>
                              <div className="flex flex-wrap gap-1">
                                {policyGuardrails[policy].map((guardrail: string, gIndex: number) => (
                                  <Badge key={gIndex} color="blue" size="xs">
                                    {guardrail}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <Text className="text-gray-500">No policies configured</Text>
                  )}
                </Card>

                <LoggingSettingsView
                  loggingConfigs={info.metadata?.logging || []}
                  disabledCallbacks={[]}
                  variant="card"
                />
              </Grid>
            ),
          },
          {
            key: TEAM_INFO_TAB_KEYS.MEMBERS,
            label: TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.MEMBERS],
            children: (
              <TeamMembersComponent
                teamData={teamData}
                canEditTeam={canEditTeam}
                handleMemberDelete={handleMemberDelete}
                setSelectedEditMember={setSelectedEditMember}
                setIsEditMemberModalVisible={setIsEditMemberModalVisible}
                setIsAddMemberModalVisible={setIsAddMemberModalVisible}
              />
            ),
          },
          {
            key: TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS,
            label: TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS],
            children: (
              <MemberPermissions teamId={teamId} accessToken={accessToken} canEditTeam={canEditTeam} />
            ),
          },
          {
            key: TEAM_INFO_TAB_KEYS.SETTINGS,
            label: TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.SETTINGS],
            children: (
              <Card className="overflow-y-auto max-h-[65vh]">
                <div className="flex justify-between items-center mb-4">
                  <Title>Team Settings</Title>
                  {canEditTeam && !isEditing && (
                    <Button icon={<EditOutlined className="h-4 w-4" />} onClick={() => setIsEditing(true)}>Edit Settings</Button>
                  )}
                </div>

                {isEditing ? (
                  <Form
                    form={form}
                    onFinish={handleTeamUpdate}
                    initialValues={{
                      ...info,
                      team_alias: info.team_alias,
                      models: info.models,
                      tpm_limit: info.tpm_limit,
                      rpm_limit: info.rpm_limit,
                      max_budget: info.max_budget,
                      soft_budget: info.soft_budget,
                      budget_duration: info.budget_duration,
                      team_member_tpm_limit: info.team_member_budget_table?.tpm_limit,
                      team_member_rpm_limit: info.team_member_budget_table?.rpm_limit,
                      team_member_budget: info.team_member_budget_table?.max_budget,
                      team_member_budget_duration: info.team_member_budget_table?.budget_duration,
                      guardrails: info.metadata?.guardrails || [],
                      policies: info.policies || [],
                      disable_global_guardrails: info.metadata?.disable_global_guardrails || false,
                      soft_budget_alerting_emails:
                        Array.isArray(info.metadata?.soft_budget_alerting_emails)
                          ? info.metadata.soft_budget_alerting_emails.join(", ")
                          : "",
                      metadata: info.metadata
                        ? JSON.stringify(
                          (({ logging, secret_manager_settings, soft_budget_alerting_emails, ...rest }) => rest)(info.metadata),
                          null,
                          2,
                        )
                        : "",
                      logging_settings: info.metadata?.logging || [],
                      secret_manager_settings: info.metadata?.secret_manager_settings
                        ? JSON.stringify(info.metadata.secret_manager_settings, null, 2)
                        : "",
                      organization_id: info.organization_id,
                      vector_stores: info.object_permission?.vector_stores || [],
                      mcp_servers: info.object_permission?.mcp_servers || [],
                      mcp_access_groups: info.object_permission?.mcp_access_groups || [],
                      mcp_servers_and_groups: {
                        servers: info.object_permission?.mcp_servers || [],
                        accessGroups: info.object_permission?.mcp_access_groups || [],
                      },
                      mcp_tool_permissions: info.object_permission?.mcp_tool_permissions || {},
                      agents_and_groups: {
                        agents: info.object_permission?.agents || [],
                        accessGroups: info.object_permission?.agent_access_groups || [],
                      },
                      access_group_ids: info.access_group_ids || [],
                    }}
                    layout="vertical"
                  >
                    <Form.Item
                      label="Team Name"
                      name="team_alias"
                      rules={[{ required: true, message: "Please input a team name" }]}
                    >
                      <Input type="" />
                    </Form.Item>

                    <Form.Item
                      label="Models"
                      name="models"
                      rules={[{ required: true, message: "Please select at least one model" }]}
                    >
                      <ModelSelect
                        value={form.getFieldValue("models") || []}
                        onChange={(values) => form.setFieldValue("models", values)}
                        teamID={teamId}
                        organizationID={teamData?.team_info?.organization_id || undefined}
                        options={{
                          includeSpecialOptions: true,
                          includeUserModels: !teamData?.team_info?.organization_id,
                          showAllProxyModelsOverride: isProxyAdminRole(userRole) && !teamData?.team_info?.organization_id,
                        }}
                        context="team"
                        dataTestId="models-select"
                      />
                    </Form.Item>

                    <Form.Item label="Max Budget (USD)" name="max_budget">
                      <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item label="Soft Budget (USD)" name="soft_budget">
                      <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item
                      label="Soft Budget Alerting Emails"
                      name="soft_budget_alerting_emails"
                      tooltip="Comma-separated email addresses to receive alerts when the soft budget is reached"
                    >
                      <Input placeholder="example1@test.com, example2@test.com" />
                    </Form.Item>

                    <Form.Item
                      label="Team Member Budget (USD)"
                      name="team_member_budget"
                      tooltip="This is the individual budget for a user in the team."
                    >
                      <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item label="Team Member Budget Duration" name="team_member_budget_duration">
                      <DurationSelect
                        onChange={(value) => form.setFieldValue("team_member_budget_duration", value)}
                        value={form.getFieldValue("team_member_budget_duration")}
                      />
                    </Form.Item>

                    <Form.Item
                      label="Team Member Key Duration (eg: 1d, 1mo)"
                      name="team_member_key_duration"
                      tooltip="Set a limit to the duration of a team member's key. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days), 1mo (month)"
                    >
                      <TextInput placeholder="e.g., 30d" />
                    </Form.Item>

                    <Form.Item
                      label="Team Member TPM Limit"
                      name="team_member_tpm_limit"
                      tooltip="Default tokens per minute limit for an individual team member. This limit applies to all requests the user makes within this team. Can be overridden per member."
                    >
                      <NumericalInput step={1} style={{ width: "100%" }} placeholder="e.g., 1000" />
                    </Form.Item>

                    <Form.Item
                      label="Team Member RPM Limit"
                      name="team_member_rpm_limit"
                      tooltip="Default requests per minute limit for an individual team member. This limit applies to all requests the user makes within this team. Can be overridden per member."
                    >
                      <NumericalInput step={1} style={{ width: "100%" }} placeholder="e.g., 100" />
                    </Form.Item>

                    <Form.Item label="Reset Budget" name="budget_duration">
                      <Select placeholder="n/a">
                        <Select.Option value="24h">daily</Select.Option>
                        <Select.Option value="7d">weekly</Select.Option>
                        <Select.Option value="30d">monthly</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                      <NumericalInput step={1} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
                      <NumericalInput step={1} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Guardrails{" "}
                          <Tooltip title="Setup your first guardrail">
                            <a
                              href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </span>
                      }
                      name="guardrails"
                      help="Select existing guardrails or enter new ones"
                    >
                      <Select
                        mode="tags"
                        placeholder="Select or enter guardrails"
                        options={guardrailsList.map((name) => ({ value: name, label: name }))}
                      />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Disable Global Guardrails
                          <Tooltip title="When enabled, this team will bypass any guardrails configured to run on every request (global guardrails)">
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="disable_global_guardrails"
                      valuePropName="checked"
                      help="Bypass global guardrails for this team"
                    >
                      <Switch checkedChildren="Yes" unCheckedChildren="No" />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Policies{" "}
                          <Tooltip title="Apply policies to this team to control guardrails and other settings">
                            <a
                              href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </span>
                      }
                      name="policies"
                      help="Select existing policies or enter new ones"
                    >
                      <Select
                        mode="tags"
                        placeholder="Select or enter policies"
                        options={policiesList.map((name) => ({ value: name, label: name }))}
                      />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Access Groups{" "}
                          <Tooltip title="Assign access groups to this team. Access groups control which models, MCP servers, and agents this team can use">
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="access_group_ids"
                    >
                      <AccessGroupSelector placeholder="Select access groups (optional)" />
                    </Form.Item>

                    <Form.Item label="Vector Stores" name="vector_stores" aria-label="Vector Stores">
                      <VectorStoreSelector
                        onChange={(values: string[]) => form.setFieldValue("vector_stores", values)}
                        value={form.getFieldValue("vector_stores")}
                        accessToken={accessToken || ""}
                        placeholder="Select vector stores"
                      />
                    </Form.Item>

                    <Form.Item label="Allowed Pass Through Routes" name="allowed_passthrough_routes">
                      <PassThroughRoutesSelector
                        onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
                        value={form.getFieldValue("allowed_passthrough_routes")}
                        accessToken={accessToken || ""}
                        placeholder="Select pass through routes"
                      />
                    </Form.Item>

                    <Form.Item label="MCP Servers / Access Groups" name="mcp_servers_and_groups">
                      <MCPServerSelector
                        onChange={(val) => form.setFieldValue("mcp_servers_and_groups", val)}
                        value={form.getFieldValue("mcp_servers_and_groups")}
                        accessToken={accessToken || ""}
                        placeholder="Select MCP servers or access groups (optional)"
                      />
                    </Form.Item>

                    {/* Hidden field to register mcp_tool_permissions with the form */}
                    <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
                      <Input type="hidden" />
                    </Form.Item>

                    <Form.Item
                      noStyle
                      shouldUpdate={(prevValues, currentValues) =>
                        prevValues.mcp_servers_and_groups !== currentValues.mcp_servers_and_groups ||
                        prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
                      }
                    >
                      {() => (
                        <div className="mb-6">
                          <MCPToolPermissions
                            accessToken={accessToken || ""}
                            selectedServers={form.getFieldValue("mcp_servers_and_groups")?.servers || []}
                            toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                            onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                          />
                        </div>
                      )}
                    </Form.Item>

                    <Form.Item label="Agents / Access Groups" name="agents_and_groups">
                      <AgentSelector
                        onChange={(val) => form.setFieldValue("agents_and_groups", val)}
                        value={form.getFieldValue("agents_and_groups")}
                        accessToken={accessToken || ""}
                        placeholder="Select agents or access groups (optional)"
                      />
                    </Form.Item>

                    <Form.Item label="Organization ID" name="organization_id">
                      <Input type="" disabled />
                    </Form.Item>

                    <Form.Item label="Logging Settings" name="logging_settings">
                      <EditLoggingSettings
                        value={form.getFieldValue("logging_settings")}
                        onChange={(values) => form.setFieldValue("logging_settings", values)}
                      />
                    </Form.Item>

                    <Form.Item
                      label="Secret Manager Settings"
                      name="secret_manager_settings"
                      help={
                        premiumUser
                          ? "Enter secret manager configuration as a JSON object."
                          : "Premium feature - Upgrade to manage secret manager settings."
                      }
                      rules={[
                        {
                          validator: async (_, value) => {
                            if (!value) {
                              return Promise.resolve();
                            }
                            try {
                              JSON.parse(value);
                              return Promise.resolve();
                            } catch (error) {
                              return Promise.reject(new Error("Please enter valid JSON"));
                            }
                          },
                        },
                      ]}
                    >
                      <Input.TextArea
                        rows={6}
                        placeholder='{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
                        disabled={!premiumUser}
                      />
                    </Form.Item>

                    <Form.Item label="Metadata" name="metadata">
                      <Input.TextArea rows={10} />
                    </Form.Item>

                    <div className="sticky z-10 bg-white p-4 pr-0 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
                      <div className="flex justify-end items-center gap-2">
                        <Button onClick={() => setIsEditing(false)} disabled={isTeamSaving}>
                          Cancel
                        </Button>
                        <Button icon={<SaveOutlined className="h-4 w-4" />} type="primary" htmlType="submit" loading={isTeamSaving}>
                          Save Changes
                        </Button>
                      </div>
                    </div>
                  </Form>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">Team Name</Text>
                      <div>{info.team_alias}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Team ID</Text>
                      <div className="font-mono">{info.team_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Created At</Text>
                      <div>{new Date(info.created_at).toLocaleString()}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Models</Text>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {info.models.map((model, index) => (
                          <Badge key={index} color="red">
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Rate Limits</Text>
                      <div>TPM: {info.tpm_limit || "Unlimited"}</div>
                      <div>RPM: {info.rpm_limit || "Unlimited"}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Team Budget</Text>
                      <div>
                        Max Budget:{" "}
                        {info.max_budget !== null ? `$${formatNumberWithCommas(info.max_budget, 4)}` : "No Limit"}
                      </div>
                      <div>
                        Soft Budget:{" "}
                        {info.soft_budget !== null && info.soft_budget !== undefined
                          ? `$${formatNumberWithCommas(info.soft_budget, 4)}`
                          : "No Limit"}
                      </div>
                      <div>Budget Reset: {info.budget_duration || "Never"}</div>
                      {info.metadata?.soft_budget_alerting_emails &&
                        Array.isArray(info.metadata.soft_budget_alerting_emails) &&
                        info.metadata.soft_budget_alerting_emails.length > 0 && (
                          <div>
                            Soft Budget Alerting Emails: {info.metadata.soft_budget_alerting_emails.join(", ")}
                          </div>
                        )}
                    </div>
                    <div>
                      <Text className="font-medium">
                        Team Member Settings{" "}
                        <Tooltip title="These are limits on individual team members">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </Text>
                      <div>Max Budget: {info.team_member_budget_table?.max_budget || "No Limit"}</div>
                      <div>Budget Duration: {info.team_member_budget_table?.budget_duration || "No Limit"}</div>
                      <div>Key Duration: {info.metadata?.team_member_key_duration || "No Limit"}</div>
                      <div>TPM Limit: {info.team_member_budget_table?.tpm_limit || "No Limit"}</div>
                      <div>RPM Limit: {info.team_member_budget_table?.rpm_limit || "No Limit"}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Organization ID</Text>
                      <div>{info.organization_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">Status</Text>
                      <Badge color={info.blocked ? "red" : "green"}>{info.blocked ? "Blocked" : "Active"}</Badge>
                    </div>

                    <div>
                      <Text className="font-medium">Disable Global Guardrails</Text>
                      <div>
                        {info.metadata?.disable_global_guardrails === true ? (
                          <Badge color="yellow">Enabled - Global guardrails bypassed</Badge>
                        ) : (
                          <Badge color="green">Disabled - Global guardrails active</Badge>
                        )}
                      </div>
                    </div>

                    <ObjectPermissionsView
                      objectPermission={info.object_permission}
                      variant="inline"
                      className="pt-4 border-t border-gray-200"
                      accessToken={accessToken}
                    />

                    <LoggingSettingsView
                      loggingConfigs={info.metadata?.logging || []}
                      disabledCallbacks={[]}
                      variant="inline"
                      className="pt-4 border-t border-gray-200"
                    />

                    {info.metadata?.secret_manager_settings && (
                      <div className="pt-4 border-t border-gray-200">
                        <Text className="font-medium">Secret Manager Settings</Text>
                        <pre className="mt-2 bg-gray-50 p-3 rounded text-xs overflow-x-auto">
                          {JSON.stringify(info.metadata.secret_manager_settings, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </Card>
            ),
          },
        ].filter(tab => visibleTabs.includes(tab.key))}
      />

      <MemberModal
        visible={isEditMemberModalVisible}
        onCancel={() => setIsEditMemberModalVisible(false)}
        onSubmit={handleMemberUpdate}
        initialData={selectedEditMember}
        mode="edit"
        config={{
          title: "Edit Member",
          showEmail: true,
          showUserId: true,
          roleOptions: [
            { label: "Admin", value: "admin" },
            { label: "User", value: "user" },
          ],
          additionalFields: [
            {
              name: "max_budget_in_team",
              label: (
                <span>
                  Team Member Budget (USD){" "}
                  <Tooltip title="Maximum amount in USD this member can spend within this team. This is separate from any global user budget limits">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 0.01,
              min: 0,
              placeholder: "Budget limit for this member within this team",
            },
            {
              name: "tpm_limit",
              label: (
                <span>
                  Team Member TPM Limit{" "}
                  <Tooltip title="Maximum tokens per minute this member can use within this team. This is separate from any global user TPM limit">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 1,
              min: 0,
              placeholder: "Tokens per minute limit for this member in this team",
            },
            {
              name: "rpm_limit",
              label: (
                <span>
                  Team Member RPM Limit{" "}
                  <Tooltip title="Maximum requests per minute this member can make within this team. This is separate from any global user RPM limit">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 1,
              min: 0,
              placeholder: "Requests per minute limit for this member in this team",
            },
          ],
        }}
      />

      <UserSearchModal
        isVisible={isAddMemberModalVisible}
        onCancel={() => setIsAddMemberModalVisible(false)}
        onSubmit={handleMemberCreate}
        accessToken={accessToken}
      />

      {/* Delete Member Confirmation Modal */}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Team Member"
        alertMessage="Removing team members will also delete any keys created by or created for this member."
        message="Are you sure you want to remove this member from the team? This action cannot be undone."
        resourceInformationTitle="Team Member Information"
        resourceInformation={[
          { label: "User ID", value: memberToDelete?.user_id, code: true },
          { label: "Email", value: memberToDelete?.user_email },
          { label: "Role", value: memberToDelete?.role },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={isDeleting}
      />
    </div>
  );
};

export default TeamInfoView;
