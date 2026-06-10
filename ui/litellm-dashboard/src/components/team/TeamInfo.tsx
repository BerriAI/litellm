import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { organizationKeys, useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useQueryClient } from "@tanstack/react-query";
import UserSearchModal from "@/components/common_components/user_search_modal";
import {
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
import { useGuardrails } from "@/app/(dashboard)/hooks/guardrails/useGuardrails";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { mapEmptyStringToNull } from "@/utils/keyUpdateUtils";
import { isProxyAdminRole } from "@/utils/roles";
import {
  EditOutlined,
  GlobalOutlined,
  InfoCircleOutlined,
  MinusCircleOutlined,
  PlusOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Accordion, AccordionBody, AccordionHeader, Badge, Card, Grid, Text, TextInput, Title } from "@tremor/react";
import { Button, Form, Input, InputNumber, Select, Space, Switch, Tabs, Tag, Tooltip } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { CheckIcon, CopyIcon } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import AgentSelector from "../agent_management/AgentSelector";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import DurationSelect from "../common_components/DurationSelect";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import { unfurlWildcardModelsInList } from "../key_team_helpers/fetch_available_models_team_key";
import GuardrailSettingsView from "../GuardrailSettingsView";
import LoggingSettingsView from "../logging_settings_view";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import { ModelSelect } from "../ModelSelect/ModelSelect";
import NotificationsManager from "../molecules/notifications_manager";
import { fetchMCPAccessGroups } from "../networking";
import ObjectPermissionsView from "../object_permissions_view";
import NumericalInput from "../shared/numerical_input";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import SearchToolSelector from "../SearchTools/SearchToolSelector";
import EditLoggingSettings from "./EditLoggingSettings";
import RouterSettingsAccordion, { RouterSettingsAccordionRef } from "../common_components/RouterSettingsAccordion";
import MemberModal from "./EditMembership";
import MemberPermissions from "./member_permissions";
import MyUserTab from "./MyUserTab";
import { getTeamInfoDefaultTab, getTeamInfoVisibleTabs, TEAM_INFO_TAB_KEYS } from "./tabVisibilityUtils";
import TeamMembersComponent from "./TeamMemberTab";
import { TeamVirtualKeysTable } from "./TeamVirtualKeysTable";

export interface TeamMembership {
  user_id: string;
  team_id: string;
  budget_id: string;
  spend: number;
  total_spend: number | null;
  litellm_budget_table: {
    budget_id: string;
    soft_budget: number | null;
    max_budget: number | null;
    max_parallel_requests: number | null;
    tpm_limit: number | null;
    rpm_limit: number | null;
    model_max_budget: Record<string, number> | null;
    budget_duration: string | null;
    budget_reset_at: string | null;
    allowed_models?: string[] | null;
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
    default_team_member_models?: string[];
    access_group_models?: string[];
    access_group_mcp_server_ids?: string[];
    access_group_agent_ids?: string[];
    router_settings?: Record<string, any>;
    guardrails?: string[];
    policies?: string[];
    object_permission?: {
      object_permission_id: string;
      mcp_servers: string[];
      mcp_access_groups?: string[];
      mcp_tool_permissions?: Record<string, string[]>;
      mcp_toolsets?: string[];
      vector_stores: string[];
      agents?: string[];
      agent_access_groups?: string[];
      search_tools?: string[];
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
  is_org_admin?: boolean;
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
  is_org_admin = false,
  userModels,
  editTeam,
  premiumUser = false,
  onUpdate,
}) => {
  const { t } = useTranslation();
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
  const { data: guardrailsData, isLoading: isGuardrailsLoading } = useGuardrails();
  const globalGuardrailNames = guardrailsData?.globalGuardrailNames ?? new Set<string>();
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [policyGuardrails, setPolicyGuardrails] = useState<Record<string, string[]>>({});
  const [loadingPolicies, setLoadingPolicies] = useState(false);
  const [memberToDelete, setMemberToDelete] = useState<Member | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isTeamSaving, setIsTeamSaving] = useState(false);
  const routerSettingsRef = React.useRef<RouterSettingsAccordionRef>(null);
  const [organization, setOrganization] = useState<Organization | null>(null);
  const { userRole, userId } = useAuthorized();
  const { data: userOrganizations = [] } = useOrganizations();
  const queryClient = useQueryClient();

  // Check if user is org admin for this team's organization
  const isOrgAdminForTeam = useMemo(() => {
    const teamOrgId = teamData?.team_info?.organization_id;
    if (!teamOrgId || !userId) return false;
    const org = userOrganizations.find((o) => o.organization_id === teamOrgId);
    return org?.members?.some((m: any) => m.user_id === userId && m.user_role === "org_admin") ?? false;
  }, [teamData, userOrganizations, userId]);

  // Models currently selected in the team edit form, used to scope the per-model
  // rate limit dropdown to models this team actually has access to.
  const selectedModelsInForm = Form.useWatch("models", form) as string[] | undefined;
  const killSwitchOn = Form.useWatch("disable_global_guardrails", form) as boolean | undefined;
  const availableRateLimitModels = useMemo(() => {
    const selected = selectedModelsInForm ?? teamData?.team_info?.models ?? [];
    if (selected.includes("all-proxy-models") || selected.includes("all-team-models")) {
      return userModels;
    }
    return unfurlWildcardModelsInList(selected, userModels);
  }, [selectedModelsInForm, teamData, userModels]);

  const canEditTeam = is_team_admin || is_proxy_admin || is_org_admin || isOrgAdminForTeam;
  const visibleTabs = useMemo(() => getTeamInfoVisibleTabs(canEditTeam), [canEditTeam]);
  const defaultTabKey = useMemo(() => getTeamInfoDefaultTab(editTeam, canEditTeam), [editTeam, canEditTeam]);

  const fetchTeamInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await teamInfoCall(accessToken, teamId);
      setTeamData(response);
    } catch (error) {
      NotificationsManager.fromBackend(t("teamPage.teamInfo.failedToLoadTeam"));
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
          }),
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

      NotificationsManager.success(t("teamPage.teamInfo.memberAddedSuccess"));
      setIsAddMemberModalVisible(false);
      form.resetFields();

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = t("teamPage.teamInfo.failedToAddMember");

      if (error?.raw?.detail?.error?.includes("Assigning team admins is a premium feature")) {
        errMsg = t("teamPage.teamInfo.assignAdminsEnterprise");
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
        budget_duration: values.budget_duration,
        allowed_models: values.allowed_models,
      };
      MessageManager.destroy(); // Remove all existing toasts

      await teamMemberUpdateCall(accessToken, teamId, member);

      NotificationsManager.success(t("teamPage.teamInfo.memberUpdatedSuccess"));
      setIsEditMemberModalVisible(false);

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error: any) {
      let errMsg = t("teamPage.teamInfo.failedToUpdateMember");
      if (error?.raw?.detail?.includes("Assigning team admins is a premium feature")) {
        errMsg = t("teamPage.teamInfo.assignAdminsEnterprise");
      } else if (error?.message) {
        errMsg = error.message;
      }
      setIsEditMemberModalVisible(false);

      MessageManager.destroy(); // Remove all existing toasts

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

      NotificationsManager.success(t("teamPage.teamInfo.memberRemovedSuccess"));

      // Fetch updated team info
      const updatedTeamData = await teamInfoCall(accessToken, teamId);
      setTeamData(updatedTeamData);

      // Notify parent component of the update
      onUpdate(updatedTeamData);
    } catch (error) {
      NotificationsManager.fromBackend(t("teamPage.teamInfo.failedToRemoveMember"));
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
        NotificationsManager.fromBackend(t("teamPage.teamInfo.invalidMetadataJson"));
        return;
      }

      let secretManagerSettings: Record<string, any> | undefined;
      if (typeof values.secret_manager_settings === "string") {
        const trimmedSecretConfig = values.secret_manager_settings.trim();
        if (trimmedSecretConfig.length > 0) {
          try {
            secretManagerSettings = JSON.parse(values.secret_manager_settings);
          } catch (e) {
            NotificationsManager.fromBackend(t("teamPage.teamInfo.invalidSecretManagerJson"));
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

      const modelTpmLimit: Record<string, number> = {};
      const modelRpmLimit: Record<string, number> = {};
      for (const entry of (values.modelLimits ?? []) as { model?: string; tpm?: number; rpm?: number }[]) {
        if (entry?.model) {
          if (entry.tpm != null) modelTpmLimit[entry.model] = entry.tpm;
          if (entry.rpm != null) modelRpmLimit[entry.model] = entry.rpm;
        }
      }

      const killSwitchOnAtSave = values.disable_global_guardrails === true;
      const optedOutGlobalGuardrails = killSwitchOnAtSave
        ? Array.from(globalGuardrailNames)
        : Array.from(globalGuardrailNames).filter((n) => !(values.guardrails || []).includes(n));

      // Non-proxy-admins can't set allowed_passthrough_routes; preserve the
      // stored value so an unrelated save can't wipe it.
      const passthroughRoutesMetadata = is_proxy_admin
        ? { allowed_passthrough_routes: values.allowed_passthrough_routes || [] }
        : info.metadata?.allowed_passthrough_routes
          ? { allowed_passthrough_routes: info.metadata.allowed_passthrough_routes }
          : {};

      const updateData: any = {
        team_id: teamId,
        team_alias: values.team_alias,
        models: values.models,
        tpm_limit: sanitizeNumeric(values.tpm_limit),
        rpm_limit: sanitizeNumeric(values.rpm_limit),
        model_tpm_limit: modelTpmLimit,
        model_rpm_limit: modelRpmLimit,
        max_budget: values.max_budget,
        soft_budget: sanitizeNumeric(values.soft_budget),
        budget_duration: values.budget_duration,
        metadata: {
          ...parsedMetadata,
          ...passthroughRoutesMetadata,
          guardrails: (values.guardrails || []).filter((n: string) => !globalGuardrailNames.has(n)),
          opted_out_global_guardrails: optedOutGlobalGuardrails,
          ...(values.logging_settings?.length > 0 ? { logging: values.logging_settings } : {}),
          disable_global_guardrails: killSwitchOnAtSave,
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
        ...(values.organization_id !== info.organization_id ? { organization_id: values.organization_id ?? null } : {}),
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
      const { servers, accessGroups, toolsets } = values.mcp_servers_and_groups || {
        servers: [],
        accessGroups: [],
        toolsets: [],
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
      if (toolsets) {
        updateData.object_permission.mcp_toolsets = toolsets;
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

      if (Array.isArray(values.object_permission_search_tools)) {
        updateData.object_permission.search_tools = values.object_permission_search_tools;
      }

      // Pass access_group_ids to the update request
      if (values.access_group_ids !== undefined) {
        updateData.access_group_ids = values.access_group_ids;
      }

      // Pass default_team_member_models to the update request
      if (values.default_team_member_models !== undefined) {
        updateData.default_team_member_models = values.default_team_member_models;
      }

      // Handle router_settings - read fresh values from DOM at save time.
      const currentRouterSettings = routerSettingsRef.current?.getValue();
      if (currentRouterSettings?.router_settings) {
        const isMeaningfulValue = (value: unknown) =>
          value !== null &&
          value !== undefined &&
          value !== "" &&
          value !== false &&
          !(Array.isArray(value) && value.length === 0);

        const hasNewValues = Object.values(currentRouterSettings.router_settings).some(isMeaningfulValue);
        const hadExistingSettings = info.router_settings && Object.values(info.router_settings).some(isMeaningfulValue);

        // Send if there are new values OR if the user is clearing existing ones
        if (hasNewValues || hadExistingSettings) {
          updateData.router_settings = currentRouterSettings.router_settings;
        }
      }

      const response = await teamUpdateCall(accessToken, updateData);
      queryClient.invalidateQueries({ queryKey: organizationKeys.all });

      NotificationsManager.success(t("teamPage.teamInfo.teamSettingsUpdatedSuccess"));
      setIsEditing(false);
      fetchTeamInfo();
    } catch (error) {
      console.error("Error updating team:", error);
    } finally {
      setIsTeamSaving(false);
    }
  };

  if (loading) {
    return <div className="p-4">{t("common.loading")}</div>;
  }

  if (!teamData?.team_info) {
    return <div className="p-4">{t("teamPage.teamInfo.teamNotFound")}</div>;
  }

  const { team_info: info } = teamData;

  const initialKillSwitchOn = info.metadata?.disable_global_guardrails === true;
  const optedOutGlobals = new Set<string>(
    Array.isArray(info.metadata?.opted_out_global_guardrails) ? info.metadata.opted_out_global_guardrails : [],
  );
  const nonGlobalOptIns: string[] = (Array.isArray(info.metadata?.guardrails) ? info.metadata.guardrails : []).filter(
    (n: string) => !globalGuardrailNames.has(n),
  );
  const effectiveGuardrails: string[] = initialKillSwitchOn
    ? nonGlobalOptIns
    : [...Array.from(globalGuardrailNames).filter((n) => !optedOutGlobals.has(n)), ...nonGlobalOptIns];

  const preventTagMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const renderGuardrailTag = ({ label, value, closable, onClose }: any) => {
    const isGlobal = globalGuardrailNames.has(value);
    return (
      <Tag
        color="blue"
        closable={closable}
        onClose={onClose}
        onMouseDown={preventTagMouseDown}
        style={{ marginInlineEnd: 4 }}
      >
        {isGlobal && (
          <GlobalOutlined style={{ marginInlineEnd: 4 }} aria-label={t("teamPage.teamInfo.globalGuardrailAriaLabel")} />
        )}
        {label}
      </Tag>
    );
  };

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
          <Button type="text" icon={<ArrowLeftIcon className="h-4 w-4" />} onClick={onClose} className="mb-4">
            {t("teamPage.teamInfo.backToTeams")}
          </Button>
          <Title>{info.team_alias}</Title>
          <div className="flex items-center">
            <Text className="text-gray-500 font-mono">{info.team_id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["team-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(info.team_id, "team-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["team-id"]
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
            label: t("teamPage.teamInfo.tabOverview"),
            children: (
              <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
                <Card>
                  <Text>{t("teamPage.teamInfo.budgetStatus")}</Text>
                  <div className="mt-2">
                    <Title>${formatNumberWithCommas(info.spend, 4)}</Title>
                    <Text>
                      {t("teamPage.teamInfo.budgetOf", {
                        value:
                          info.max_budget === null
                            ? t("teamPage.teamInfo.unlimited")
                            : `$${formatNumberWithCommas(info.max_budget, 4)}`,
                      })}
                    </Text>
                    {info.budget_duration && (
                      <Text className="text-gray-500">
                        {t("teamPage.teamInfo.budgetReset", { duration: info.budget_duration })}
                      </Text>
                    )}
                    <br />
                    {info.team_member_budget_table && (
                      <Text className="text-gray-500">
                        {t("teamPage.teamInfo.teamMemberBudget", {
                          amount: formatNumberWithCommas(info.team_member_budget_table.max_budget, 4),
                        })}
                      </Text>
                    )}
                  </div>
                </Card>

                <Card>
                  <Text>{t("teamPage.teamInfo.rateLimits")}</Text>
                  <div className="mt-2">
                    <Text>
                      {t("teamPage.teamInfo.tpmLabel", {
                        value: info.tpm_limit || t("teamPage.teamInfo.unlimited"),
                      })}
                    </Text>
                    <Text>
                      {t("teamPage.teamInfo.rpmLabel", {
                        value: info.rpm_limit || t("teamPage.teamInfo.unlimited"),
                      })}
                    </Text>
                    {info.max_parallel_requests && (
                      <Text>{t("teamPage.teamInfo.maxParallelRequests", { value: info.max_parallel_requests })}</Text>
                    )}
                    {(() => {
                      const modelTpm = (info.metadata?.model_tpm_limit ?? {}) as Record<string, number>;
                      const modelRpm = (info.metadata?.model_rpm_limit ?? {}) as Record<string, number>;
                      const models = Array.from(new Set([...Object.keys(modelTpm), ...Object.keys(modelRpm)]));
                      if (models.length === 0) return null;
                      return (
                        <div className="mt-3">
                          <Text className="text-gray-500">{t("teamPage.teamInfo.perModelLimits")}</Text>
                          {models.map((m) => (
                            <Text key={m} className="text-xs">
                              {t("teamPage.teamInfo.perModelEntry", {
                                model: m,
                                tpm: modelTpm[m] ?? "—",
                                rpm: modelRpm[m] ?? "—",
                              })}
                            </Text>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                </Card>

                <Card>
                  <Text>{t("teamPage.teamInfo.modelsLabel")}</Text>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {info.models.length === 0 || info.models.includes("all-proxy-models") ? (
                      <Badge color="red">{t("teamPage.teamInfo.allProxyModels")}</Badge>
                    ) : (
                      <>
                        {info.models.map((model: string, index: number) => (
                          <Badge key={`direct-${index}`} color="blue">
                            {model}
                          </Badge>
                        ))}
                        {(info.access_group_models || []).map((model: string, index: number) => (
                          <Badge key={`ag-${index}`} color="green" title={t("teamPage.teamInfo.fromAccessGroup")}>
                            {model}
                          </Badge>
                        ))}
                      </>
                    )}
                  </div>
                </Card>

                <Card>
                  <Text className="font-semibold text-gray-900">{t("teamPage.teamInfo.virtualKeys")}</Text>
                  <div className="mt-2">
                    <Text>
                      {t("teamPage.teamInfo.userKeys", { count: teamData.keys.filter((key) => key.user_id).length })}
                    </Text>
                    <Text>
                      {t("teamPage.teamInfo.serviceAccountKeys", {
                        count: teamData.keys.filter((key) => !key.user_id).length,
                      })}
                    </Text>
                    <Text className="text-gray-500">
                      {t("teamPage.teamInfo.totalKeys", { count: teamData.keys.length })}
                    </Text>
                  </div>
                </Card>

                <ObjectPermissionsView
                  objectPermission={info.object_permission}
                  variant="card"
                  accessToken={accessToken}
                />

                <Card>
                  <GuardrailSettingsView
                    globalGuardrailNames={globalGuardrailNames}
                    teamGuardrails={Array.isArray(info.metadata?.guardrails) ? info.metadata.guardrails : []}
                    optedOutGlobalGuardrails={
                      Array.isArray(info.metadata?.opted_out_global_guardrails)
                        ? info.metadata.opted_out_global_guardrails
                        : []
                    }
                    killSwitchOn={initialKillSwitchOn}
                    variant="inline"
                  />
                </Card>

                <Card>
                  <Text className="font-semibold text-gray-900 mb-3">{t("teamPage.teamInfo.policiesLabel")}</Text>
                  {info.policies && info.policies.length > 0 ? (
                    <div className="space-y-4">
                      {info.policies.map((policy: string, index: number) => (
                        <div key={index} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Badge color="purple">{policy}</Badge>
                            {loadingPolicies && (
                              <Text className="text-xs text-gray-400">{t("teamPage.teamInfo.loadingGuardrails")}</Text>
                            )}
                          </div>
                          {!loadingPolicies && policyGuardrails[policy] && policyGuardrails[policy].length > 0 && (
                            <div className="ml-4 pl-3 border-l-2 border-gray-200">
                              <Text className="text-xs text-gray-500 mb-1">
                                {t("teamPage.teamInfo.resolvedGuardrails")}
                              </Text>
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
                    <Text className="text-gray-500">{t("teamPage.teamInfo.noPoliciesConfigured")}</Text>
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
            key: TEAM_INFO_TAB_KEYS.MY_USER,
            label: t("teamPage.teamInfo.tabMyUser"),
            children: <MyUserTab teamId={teamId} />,
          },
          {
            key: TEAM_INFO_TAB_KEYS.VIRTUAL_KEYS,
            label: t("teamPage.teamInfo.tabVirtualKeys"),
            children: <TeamVirtualKeysTable teamId={teamId} teamAlias={info.team_alias} organization={organization} />,
          },
          {
            key: TEAM_INFO_TAB_KEYS.MEMBERS,
            label: t("teamPage.teamInfo.tabMembers"),
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
            label: t("teamPage.teamInfo.tabMemberPermissions"),
            children: <MemberPermissions teamId={teamId} accessToken={accessToken} canEditTeam={canEditTeam} />,
          },
          {
            key: TEAM_INFO_TAB_KEYS.SETTINGS,
            label: t("teamPage.teamInfo.tabSettings"),
            children: (
              <Card className="overflow-y-auto max-h-[65vh]">
                <div className="flex justify-between items-center mb-4">
                  <Title>{t("teamPage.teamInfo.teamSettings")}</Title>
                  {canEditTeam && !isEditing && (
                    <Button icon={<EditOutlined className="h-4 w-4" />} onClick={() => setIsEditing(true)}>
                      {t("teamPage.teamInfo.editSettings")}
                    </Button>
                  )}
                </div>

                {isEditing && isGuardrailsLoading ? (
                  <div className="p-4">{t("common.loading")}</div>
                ) : isEditing ? (
                  <Form
                    form={form}
                    onFinish={handleTeamUpdate}
                    onValuesChange={(changedValues) => {
                      if ("disable_global_guardrails" in changedValues) {
                        const checked = changedValues.disable_global_guardrails === true;
                        const current = (form.getFieldValue("guardrails") || []) as string[];
                        const nonGlobals = current.filter((n) => !globalGuardrailNames.has(n));
                        form.setFieldValue(
                          "guardrails",
                          checked ? nonGlobals : [...Array.from(globalGuardrailNames), ...nonGlobals],
                        );
                      }
                    }}
                    initialValues={{
                      ...info,
                      team_alias: info.team_alias,
                      models: info.models,
                      tpm_limit: info.tpm_limit,
                      rpm_limit: info.rpm_limit,
                      object_permission_search_tools: info.object_permission?.search_tools || [],
                      modelLimits: Array.from(
                        new Set([
                          ...Object.keys(info.metadata?.model_tpm_limit ?? {}),
                          ...Object.keys(info.metadata?.model_rpm_limit ?? {}),
                        ]),
                      ).map((model) => ({
                        model,
                        tpm: info.metadata?.model_tpm_limit?.[model],
                        rpm: info.metadata?.model_rpm_limit?.[model],
                      })),
                      max_budget: info.max_budget,
                      soft_budget: info.soft_budget,
                      budget_duration: info.budget_duration,
                      team_member_tpm_limit: info.team_member_budget_table?.tpm_limit,
                      team_member_rpm_limit: info.team_member_budget_table?.rpm_limit,
                      team_member_budget: info.team_member_budget_table?.max_budget,
                      team_member_budget_duration: info.team_member_budget_table?.budget_duration,
                      guardrails: effectiveGuardrails,
                      policies: info.policies || [],
                      disable_global_guardrails: info.metadata?.disable_global_guardrails || false,
                      soft_budget_alerting_emails: Array.isArray(info.metadata?.soft_budget_alerting_emails)
                        ? info.metadata.soft_budget_alerting_emails.join(", ")
                        : "",
                      metadata: info.metadata
                        ? JSON.stringify(
                            (({
                              logging,
                              secret_manager_settings,
                              soft_budget_alerting_emails,
                              model_tpm_limit,
                              model_rpm_limit,
                              allowed_passthrough_routes,
                              ...rest
                            }) => rest)(info.metadata),
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
                        toolsets: info.object_permission?.mcp_toolsets || [],
                      },
                      mcp_tool_permissions: info.object_permission?.mcp_tool_permissions || {},
                      agents_and_groups: {
                        agents: info.object_permission?.agents || [],
                        accessGroups: info.object_permission?.agent_access_groups || [],
                      },
                      access_group_ids: info.access_group_ids || [],
                      default_team_member_models: info.default_team_member_models || [],
                      allowed_passthrough_routes: info.metadata?.allowed_passthrough_routes || [],
                    }}
                    layout="vertical"
                  >
                    <Form.Item
                      label={t("teamPage.teamInfo.teamNameLabel")}
                      name="team_alias"
                      rules={[{ required: true, message: t("teamPage.teamInfo.teamNameRequired") }]}
                    >
                      <Input type="" />
                    </Form.Item>

                    <Form.Item
                      label={t("teamPage.teamInfo.modelsLabel")}
                      name="models"
                      rules={[{ required: true, message: t("teamPage.teamInfo.modelsRequired") }]}
                    >
                      <ModelSelect
                        value={form.getFieldValue("models") || []}
                        onChange={(values) => form.setFieldValue("models", values)}
                        teamID={teamId}
                        organizationID={teamData?.team_info?.organization_id || undefined}
                        options={{
                          includeSpecialOptions: true,
                          includeUserModels: !teamData?.team_info?.organization_id,
                          showAllProxyModelsOverride:
                            isProxyAdminRole(userRole) && !teamData?.team_info?.organization_id,
                        }}
                        context="team"
                        dataTestId="models-select"
                      />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.maxBudgetLabel")} name="max_budget">
                      <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.softBudgetLabel")} name="soft_budget">
                      <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item
                      label={t("teamPage.teamInfo.softBudgetEmailsLabel")}
                      name="soft_budget_alerting_emails"
                      tooltip={t("teamPage.teamInfo.softBudgetEmailsTooltip")}
                    >
                      <Input placeholder={t("teamPage.teamInfo.softBudgetEmailsPlaceholder")} />
                    </Form.Item>

                    <Accordion className="mt-4 mb-4">
                      <AccordionHeader>
                        <b>{t("teamPage.teamInfo.teamMemberSettings")}</b>
                      </AccordionHeader>
                      <AccordionBody>
                        <Text className="text-xs text-gray-500 mb-4">
                          {t("teamPage.teamInfo.teamMemberSettingsDesc")}
                        </Text>
                        <Form.Item
                          label={
                            <span>
                              {t("teamPage.teamInfo.defaultModelAccessLabel")}{" "}
                              <Tooltip title={t("teamPage.teamInfo.defaultModelAccessTooltip")}>
                                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                              </Tooltip>
                            </span>
                          }
                          name="default_team_member_models"
                        >
                          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.models !== cur.models}>
                            {({ getFieldValue }) => {
                              const teamModels = getFieldValue("models") || info.models || [];
                              return (
                                <Select
                                  mode="multiple"
                                  placeholder={t("teamPage.teamInfo.defaultModelAccessPlaceholder")}
                                  value={form.getFieldValue("default_team_member_models") || []}
                                  onChange={(values) => form.setFieldValue("default_team_member_models", values)}
                                  options={teamModels.map((m: string) => ({ label: m, value: m }))}
                                />
                              );
                            }}
                          </Form.Item>
                        </Form.Item>
                        <Form.Item
                          label={t("teamPage.teamInfo.defaultBudgetLabel")}
                          name="team_member_budget"
                          tooltip={t("teamPage.teamInfo.defaultBudgetTooltip")}
                        >
                          <NumericalInput step={0.01} precision={2} style={{ width: "100%" }} />
                        </Form.Item>
                        <Form.Item
                          label={t("teamPage.teamInfo.defaultBudgetDurationLabel")}
                          name="team_member_budget_duration"
                        >
                          <DurationSelect
                            onChange={(value) => form.setFieldValue("team_member_budget_duration", value)}
                            value={form.getFieldValue("team_member_budget_duration")}
                          />
                        </Form.Item>
                        <Form.Item
                          label={t("teamPage.teamInfo.defaultKeyDurationLabel")}
                          name="team_member_key_duration"
                          tooltip={t("teamPage.teamInfo.defaultKeyDurationTooltip")}
                        >
                          <TextInput placeholder={t("teamPage.teamInfo.defaultKeyDurationPlaceholder")} />
                        </Form.Item>
                        <Form.Item
                          label={t("teamPage.teamInfo.defaultTpmLimitLabel")}
                          name="team_member_tpm_limit"
                          tooltip={t("teamPage.teamInfo.defaultTpmLimitTooltip")}
                        >
                          <NumericalInput
                            step={1}
                            style={{ width: "100%" }}
                            placeholder={t("teamPage.teamInfo.defaultTpmLimitPlaceholder")}
                          />
                        </Form.Item>
                        <Form.Item
                          label={t("teamPage.teamInfo.defaultRpmLimitLabel")}
                          name="team_member_rpm_limit"
                          tooltip={t("teamPage.teamInfo.defaultRpmLimitTooltip")}
                        >
                          <NumericalInput
                            step={1}
                            style={{ width: "100%" }}
                            placeholder={t("teamPage.teamInfo.defaultRpmLimitPlaceholder")}
                          />
                        </Form.Item>
                      </AccordionBody>
                    </Accordion>

                    <Form.Item label={t("teamPage.teamInfo.resetBudgetLabel")} name="budget_duration">
                      <Select placeholder="n/a">
                        <Select.Option value="24h">{t("teamPage.teamInfo.daily")}</Select.Option>
                        <Select.Option value="7d">{t("teamPage.teamInfo.weekly")}</Select.Option>
                        <Select.Option value="30d">{t("teamPage.teamInfo.monthly")}</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.tpmLimitLabel")} name="tpm_limit">
                      <NumericalInput step={1} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.rpmLimitLabel")} name="rpm_limit">
                      <NumericalInput step={1} style={{ width: "100%" }} />
                    </Form.Item>

                    <Form.Item
                      label={t("teamPage.teamInfo.modelSpecificRateLimitsLabel")}
                      tooltip={t("teamPage.teamInfo.modelSpecificRateLimitsTooltip")}
                    >
                      <Form.List name="modelLimits">
                        {(fields, { add, remove }) => (
                          <>
                            {fields.map(({ key, name, ...restField }) => (
                              <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                                <Form.Item
                                  {...restField}
                                  name={[name, "model"]}
                                  rules={[
                                    { required: true, message: t("teamPage.teamInfo.missingModel") },
                                    {
                                      validator: (_, value) => {
                                        if (!value) return Promise.resolve();
                                        const all = form.getFieldValue("modelLimits") ?? [];
                                        const dupes = all.filter((entry: { model?: string }) => entry?.model === value);
                                        if (dupes.length > 1) {
                                          return Promise.reject(new Error(t("teamPage.teamInfo.duplicateModel")));
                                        }
                                        return Promise.resolve();
                                      },
                                    },
                                  ]}
                                  style={{ minWidth: 240 }}
                                >
                                  <Select
                                    showSearch
                                    placeholder={t("teamPage.teamInfo.selectModelPlaceholder")}
                                    allowClear
                                    options={availableRateLimitModels.map((m) => ({
                                      value: m,
                                      label: m,
                                    }))}
                                  />
                                </Form.Item>
                                <Form.Item
                                  {...restField}
                                  name={[name, "tpm"]}
                                  rules={[
                                    {
                                      validator: async (_, value) => {
                                        const row = (form.getFieldValue("modelLimits") ?? [])[name] ?? {};
                                        if (row.model && value == null && row.rpm == null) {
                                          return Promise.reject(
                                            new Error(t("teamPage.teamInfo.setAtLeastOneLimitError")),
                                          );
                                        }
                                        return Promise.resolve();
                                      },
                                    },
                                  ]}
                                >
                                  <InputNumber placeholder={t("teamPage.teamInfo.tpmLimitPlaceholder")} min={0} />
                                </Form.Item>
                                <Form.Item {...restField} name={[name, "rpm"]}>
                                  <InputNumber placeholder={t("teamPage.teamInfo.rpmLimitPlaceholder")} min={0} />
                                </Form.Item>
                                <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ef4444" }} />
                              </Space>
                            ))}
                            <Form.Item>
                              <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                {t("teamPage.teamInfo.addModelLimitButton")}
                              </Button>
                            </Form.Item>
                          </>
                        )}
                      </Form.List>
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.routerSettingsLabel")}>
                      <RouterSettingsAccordion
                        ref={routerSettingsRef}
                        accessToken={accessToken || ""}
                        value={info.router_settings ? { router_settings: info.router_settings } : undefined}
                      />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          {t("teamPage.teamInfo.guardrailsLabel")}{" "}
                          <Tooltip title={t("teamPage.teamInfo.guardrailsTooltip")}>
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
                    >
                      <Select
                        mode="multiple"
                        placeholder={t("teamPage.teamInfo.selectGuardrailsPlaceholder")}
                        optionLabelProp="label"
                        tagRender={renderGuardrailTag}
                      >
                        <Select.OptGroup
                          label={
                            <>
                              <GlobalOutlined style={{ marginInlineEnd: 4 }} />
                              {t("teamPage.teamInfo.globalGuardrailsGroup")}
                            </>
                          }
                        >
                          {(guardrailsData?.guardrails ?? [])
                            .filter((g) => g.litellm_params?.default_on)
                            .map((g) => (
                              <Select.Option
                                key={g.guardrail_name}
                                value={g.guardrail_name}
                                label={g.guardrail_name}
                                disabled={killSwitchOn}
                              >
                                {g.guardrail_name}
                              </Select.Option>
                            ))}
                        </Select.OptGroup>
                        <Select.OptGroup label={t("teamPage.teamInfo.otherGuardrailsGroup")}>
                          {(guardrailsData?.guardrails ?? [])
                            .filter((g) => !g.litellm_params?.default_on)
                            .map((g) => (
                              <Select.Option key={g.guardrail_name} value={g.guardrail_name} label={g.guardrail_name}>
                                {g.guardrail_name}
                              </Select.Option>
                            ))}
                        </Select.OptGroup>
                      </Select>
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          {t("teamPage.teamInfo.disableGlobalGuardrailsLabel")}{" "}
                          <Tooltip title={t("teamPage.teamInfo.disableGlobalGuardrailsTooltip")}>
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="disable_global_guardrails"
                      valuePropName="checked"
                    >
                      <Switch checkedChildren={t("common.yes")} unCheckedChildren={t("common.no")} />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          {t("teamPage.teamInfo.policiesLabel")}{" "}
                          <Tooltip title={t("teamPage.teamInfo.policiesTooltip")}>
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
                    >
                      <Select
                        mode="tags"
                        placeholder={t("teamPage.teamInfo.selectPoliciesPlaceholder")}
                        options={policiesList.map((name) => ({ value: name, label: name }))}
                      />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          {t("teamPage.teamInfo.accessGroupsLabel")}{" "}
                          <Tooltip title={t("teamPage.teamInfo.accessGroupsTooltip")}>
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="access_group_ids"
                    >
                      <AccessGroupSelector placeholder={t("teamPage.teamInfo.accessGroupsPlaceholder")} />
                    </Form.Item>

                    <Form.Item
                      label={t("teamPage.teamInfo.vectorStoresLabel")}
                      name="vector_stores"
                      aria-label={t("teamPage.teamInfo.vectorStoresLabel")}
                    >
                      <VectorStoreSelector
                        onChange={(values: string[]) => form.setFieldValue("vector_stores", values)}
                        value={form.getFieldValue("vector_stores")}
                        accessToken={accessToken || ""}
                        placeholder={t("teamPage.teamInfo.vectorStoresPlaceholder")}
                      />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.allowedPassThroughLabel")} name="allowed_passthrough_routes">
                      <Tooltip
                        title={
                          !premiumUser
                            ? t("teamPage.teamInfo.passThroughPremiumTooltip")
                            : !is_proxy_admin
                              ? t("teamPage.teamInfo.passThroughAdminTooltip")
                              : ""
                        }
                        placement="top"
                      >
                        <PassThroughRoutesSelector
                          onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
                          value={form.getFieldValue("allowed_passthrough_routes")}
                          accessToken={accessToken || ""}
                          placeholder={t("teamPage.teamInfo.passThroughPlaceholder")}
                          disabled={!premiumUser || !is_proxy_admin}
                        />
                      </Tooltip>
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.mcpServersLabel")} name="mcp_servers_and_groups">
                      <MCPServerSelector
                        onChange={(val) => form.setFieldValue("mcp_servers_and_groups", val)}
                        value={form.getFieldValue("mcp_servers_and_groups")}
                        accessToken={accessToken || ""}
                        placeholder={t("teamPage.teamInfo.mcpServersPlaceholder")}
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

                    <Form.Item label={t("teamPage.teamInfo.agentsLabel")} name="agents_and_groups">
                      <AgentSelector
                        onChange={(val) => form.setFieldValue("agents_and_groups", val)}
                        value={form.getFieldValue("agents_and_groups")}
                        accessToken={accessToken || ""}
                        placeholder={t("teamPage.teamInfo.agentsPlaceholder")}
                      />
                    </Form.Item>

                    <Accordion className="mt-4 mb-4">
                      <AccordionHeader>
                        <b>{t("teamPage.teamInfo.searchToolSettingsHeader")}</b>
                      </AccordionHeader>
                      <AccordionBody>
                        <Form.Item
                          label={t("teamPage.teamInfo.allowedSearchToolsLabel")}
                          name="object_permission_search_tools"
                          tooltip={t("teamPage.teamInfo.allowedSearchToolsTooltip")}
                        >
                          <SearchToolSelector
                            onChange={(vals: string[]) => form.setFieldValue("object_permission_search_tools", vals)}
                            value={form.getFieldValue("object_permission_search_tools")}
                            accessToken={accessToken || ""}
                            placeholder={t("teamPage.teamInfo.searchToolsPlaceholder")}
                          />
                        </Form.Item>
                      </AccordionBody>
                    </Accordion>

                    <Form.Item label={t("teamPage.teamInfo.organizationLabel")} name="organization_id">
                      <Select
                        allowClear
                        placeholder={t("teamPage.teamInfo.organizationPlaceholder")}
                        showSearch
                        optionFilterProp="label"
                        options={userOrganizations.map((org) => ({
                          value: org.organization_id,
                          label: org.organization_alias || org.organization_id,
                        }))}
                      />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.loggingSettingsLabel")} name="logging_settings">
                      <EditLoggingSettings
                        value={form.getFieldValue("logging_settings")}
                        onChange={(values) => form.setFieldValue("logging_settings", values)}
                      />
                    </Form.Item>

                    <Form.Item
                      label={t("teamPage.teamInfo.secretManagerSettingsLabel")}
                      name="secret_manager_settings"
                      help={
                        premiumUser
                          ? t("teamPage.teamInfo.secretManagerSettingsHelp")
                          : t("teamPage.teamInfo.secretManagerSettingsPremiumHelp")
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
                              return Promise.reject(new Error(t("teamPage.teamInfo.secretManagerSettingsJsonError")));
                            }
                          },
                        },
                      ]}
                    >
                      <Input.TextArea
                        rows={6}
                        placeholder={t("teamPage.teamInfo.vaultConfigPlaceholder")}
                        disabled={!premiumUser}
                      />
                    </Form.Item>

                    <Form.Item label={t("teamPage.teamInfo.metadataLabel")} name="metadata">
                      <Input.TextArea rows={10} />
                    </Form.Item>

                    <div className="sticky z-10 bg-white p-4 pr-0 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
                      <div className="flex justify-end items-center gap-2">
                        <Button onClick={() => setIsEditing(false)} disabled={isTeamSaving}>
                          {t("common.cancel")}
                        </Button>
                        <Button
                          icon={<SaveOutlined className="h-4 w-4" />}
                          type="primary"
                          htmlType="submit"
                          loading={isTeamSaving}
                        >
                          {t("teamPage.teamInfo.saveChanges")}
                        </Button>
                      </div>
                    </div>
                  </Form>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.teamNameLabel")}</Text>
                      <div>{info.team_alias}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.teamIdLabel")}</Text>
                      <div className="font-mono">{info.team_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("common.createdAt")}</Text>
                      <div>{new Date(info.created_at).toLocaleString()}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.modelsLabel")}</Text>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {info.models.map((model, index) => (
                          <Badge key={index} color="red">
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    {info.default_team_member_models && info.default_team_member_models.length > 0 && (
                      <div>
                        <Text className="font-medium">{t("teamPage.teamInfo.defaultMemberModels")}</Text>
                        <div className="flex flex-wrap gap-2 mt-1">
                          {info.default_team_member_models.map((model, index) => (
                            <Badge key={index} color="blue">
                              {model}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.rateLimits")}</Text>
                      <div>
                        {t("teamPage.teamInfo.tpmLabel", {
                          value: info.tpm_limit || t("teamPage.teamInfo.unlimited"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.rpmLabel", {
                          value: info.rpm_limit || t("teamPage.teamInfo.unlimited"),
                        })}
                      </div>
                      {(() => {
                        const modelTpm = (info.metadata?.model_tpm_limit ?? {}) as Record<string, number>;
                        const modelRpm = (info.metadata?.model_rpm_limit ?? {}) as Record<string, number>;
                        const models = Array.from(new Set([...Object.keys(modelTpm), ...Object.keys(modelRpm)]));
                        if (models.length === 0) return null;
                        return (
                          <div className="mt-2">
                            <Text className="text-gray-500">{t("teamPage.teamInfo.perModelLimits")}</Text>
                            {models.map((m) => (
                              <div key={m} className="text-xs ml-2">
                                {t("teamPage.teamInfo.perModelEntry", {
                                  model: m,
                                  tpm: modelTpm[m] ?? "—",
                                  rpm: modelRpm[m] ?? "—",
                                })}
                              </div>
                            ))}
                          </div>
                        );
                      })()}
                    </div>
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.teamBudgetLabel")}</Text>
                      <div>
                        {t("teamPage.teamInfo.maxBudgetDisplay", {
                          value:
                            info.max_budget !== null
                              ? `$${formatNumberWithCommas(info.max_budget, 4)}`
                              : t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.softBudgetDisplay", {
                          value:
                            info.soft_budget !== null && info.soft_budget !== undefined
                              ? `$${formatNumberWithCommas(info.soft_budget, 4)}`
                              : t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.budgetResetDisplay", {
                          value: info.budget_duration || t("common.never"),
                        })}
                      </div>
                      {info.metadata?.soft_budget_alerting_emails &&
                        Array.isArray(info.metadata.soft_budget_alerting_emails) &&
                        info.metadata.soft_budget_alerting_emails.length > 0 && (
                          <div>
                            {t("teamPage.teamInfo.softBudgetEmailsDisplay", {
                              emails: info.metadata.soft_budget_alerting_emails.join(", "),
                            })}
                          </div>
                        )}
                    </div>
                    <div>
                      <Text className="font-medium">
                        {t("teamPage.teamInfo.teamMemberSettings")}{" "}
                        <Tooltip title={t("teamPage.teamInfo.teamMemberSettingsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </Text>
                      <div>
                        {t("teamPage.teamInfo.maxBudgetEntry", {
                          value: info.team_member_budget_table?.max_budget || t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.budgetDurationEntry", {
                          value: info.team_member_budget_table?.budget_duration || t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.keyDurationEntry", {
                          value: info.metadata?.team_member_key_duration || t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.tpmLimitEntry", {
                          value: info.team_member_budget_table?.tpm_limit || t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                      <div>
                        {t("teamPage.teamInfo.rpmLimitEntry", {
                          value: info.team_member_budget_table?.rpm_limit || t("teamPage.teamInfo.noLimit"),
                        })}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.routerSettingsLabel")}</Text>
                      {info.router_settings &&
                      Object.values(info.router_settings).some(
                        (v) => v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0),
                      ) ? (
                        <div className="mt-1 space-y-1">
                          {info.router_settings.routing_strategy && (
                            <div>
                              {t("teamPage.teamInfo.routingStrategy")}{" "}
                              <Badge color="blue">{info.router_settings.routing_strategy}</Badge>
                            </div>
                          )}
                          {info.router_settings.num_retries != null && (
                            <div>{t("teamPage.teamInfo.numRetries", { value: info.router_settings.num_retries })}</div>
                          )}
                          {info.router_settings.allowed_fails != null && (
                            <div>
                              {t("teamPage.teamInfo.allowedFailures", { value: info.router_settings.allowed_fails })}
                            </div>
                          )}
                          {info.router_settings.cooldown_time != null && (
                            <div>
                              {t("teamPage.teamInfo.cooldownTime", { value: info.router_settings.cooldown_time })}
                            </div>
                          )}
                          {info.router_settings.timeout != null && (
                            <div>{t("teamPage.teamInfo.timeout", { value: info.router_settings.timeout })}</div>
                          )}
                          {info.router_settings.retry_after != null && (
                            <div>{t("teamPage.teamInfo.retryAfter", { value: info.router_settings.retry_after })}</div>
                          )}
                          {info.router_settings.fallbacks &&
                            Array.isArray(info.router_settings.fallbacks) &&
                            info.router_settings.fallbacks.length > 0 && (
                              <div>
                                {t("teamPage.teamInfo.fallbacksCount", {
                                  count: info.router_settings.fallbacks.length,
                                })}
                              </div>
                            )}
                          {info.router_settings.enable_tag_filtering && (
                            <div>{t("teamPage.teamInfo.tagFilteringEnabled")}</div>
                          )}
                        </div>
                      ) : (
                        <div className="text-gray-400">{t("teamPage.teamInfo.noRouterSettings")}</div>
                      )}
                    </div>
                    <div>
                      <Text className="font-medium">{t("teamPage.teamInfo.organizationIdLabel")}</Text>
                      <div>{info.organization_id}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("common.status")}</Text>
                      <Badge color={info.blocked ? "red" : "green"}>
                        {info.blocked ? t("teamPage.teamInfo.blockedStatus") : t("teamPage.teamInfo.activeStatus")}
                      </Badge>
                    </div>

                    <ObjectPermissionsView
                      objectPermission={info.object_permission}
                      variant="inline"
                      className="pt-4 border-t border-gray-200"
                      accessToken={accessToken}
                    />

                    <GuardrailSettingsView
                      globalGuardrailNames={globalGuardrailNames}
                      teamGuardrails={Array.isArray(info.metadata?.guardrails) ? info.metadata.guardrails : []}
                      optedOutGlobalGuardrails={
                        Array.isArray(info.metadata?.opted_out_global_guardrails)
                          ? info.metadata.opted_out_global_guardrails
                          : []
                      }
                      killSwitchOn={initialKillSwitchOn}
                      variant="inline"
                      className="pt-4 border-t border-gray-200"
                    />

                    <LoggingSettingsView
                      loggingConfigs={info.metadata?.logging || []}
                      disabledCallbacks={[]}
                      variant="inline"
                      className="pt-4 border-t border-gray-200"
                    />

                    {info.metadata?.secret_manager_settings && (
                      <div className="pt-4 border-t border-gray-200">
                        <Text className="font-medium">{t("teamPage.teamInfo.secretManagerSettingsLabel")}</Text>
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
        ].filter((tab) => visibleTabs.includes(tab.key))}
      />

      <MemberModal
        visible={isEditMemberModalVisible}
        onCancel={() => setIsEditMemberModalVisible(false)}
        onSubmit={handleMemberUpdate}
        initialData={selectedEditMember}
        mode="edit"
        config={{
          title: t("teamPage.teamInfo.editMemberTitle"),
          showEmail: true,
          showUserId: true,
          roleOptions: [
            { label: t("teamPage.teamInfo.adminRole"), value: "admin" },
            { label: t("teamPage.teamInfo.userRole"), value: "user" },
          ],
          additionalFields: [
            {
              name: "max_budget_in_team",
              label: (
                <span>
                  {t("teamPage.teamInfo.teamMemberBudgetLabel")}{" "}
                  <Tooltip title={t("teamPage.teamInfo.teamMemberBudgetTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 0.01,
              min: 0,
              placeholder: t("teamPage.teamInfo.teamMemberBudgetPlaceholder"),
            },
            {
              name: "budget_duration",
              label: (
                <span>
                  {t("teamPage.teamInfo.budgetResetPeriodLabel")}{" "}
                  <Tooltip title={t("teamPage.teamInfo.budgetResetPeriodTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "budget-duration" as const,
            },
            {
              name: "tpm_limit",
              label: (
                <span>
                  {t("teamPage.teamInfo.teamMemberTpmLabel")}{" "}
                  <Tooltip title={t("teamPage.teamInfo.teamMemberTpmTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 1,
              min: 0,
              placeholder: t("teamPage.teamInfo.teamMemberTpmPlaceholder"),
            },
            {
              name: "rpm_limit",
              label: (
                <span>
                  {t("teamPage.teamInfo.teamMemberRpmLabel")}{" "}
                  <Tooltip title={t("teamPage.teamInfo.teamMemberRpmTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "numerical" as const,
              step: 1,
              min: 0,
              placeholder: t("teamPage.teamInfo.teamMemberRpmPlaceholder"),
            },
            {
              name: "allowed_models",
              label: (
                <span>
                  {t("teamPage.teamInfo.allowedModelsLabel")}{" "}
                  <Tooltip title={t("teamPage.teamInfo.allowedModelsTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              ),
              type: "multi-select" as const,
              options: (info.models || []).map((m: string) => ({ label: m, value: m })),
              placeholder: t("teamPage.teamInfo.allowedModelsPlaceholder"),
            },
          ],
        }}
      />

      <UserSearchModal
        isVisible={isAddMemberModalVisible}
        onCancel={() => setIsAddMemberModalVisible(false)}
        onSubmit={handleMemberCreate}
        accessToken={accessToken}
        teamId={teamId}
      />

      {/* Delete Member Confirmation Modal */}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={t("teamPage.teamInfo.deleteMemberTitle")}
        alertMessage={t("teamPage.teamInfo.deleteMemberAlert")}
        message={t("teamPage.teamInfo.deleteMemberMessage")}
        resourceInformationTitle={t("teamPage.teamInfo.deleteMemberInfoTitle")}
        resourceInformation={[
          { label: t("teamPage.teamInfo.userIdLabel"), value: memberToDelete?.user_id, code: true },
          { label: t("teamPage.teamInfo.emailLabel"), value: memberToDelete?.user_email },
          { label: t("teamPage.teamInfo.roleLabel"), value: memberToDelete?.role },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={isDeleting}
      />
    </div>
  );
};

export default TeamInfoView;
