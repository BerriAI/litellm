import { Page, PageTabs, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import useCan from "@/app/(dashboard)/hooks/useCan";
import AvailableTeamsPanel from "@/components/team/AvailableTeamsPanel";
import TeamInfoView from "@/components/team/TeamInfo";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isProxyAdminRole } from "@/utils/roles";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input as UIInput } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { labelWithDocsHint, labelWithHint } from "@/components/shared/form/LabelWithHint";
import { useZodForm } from "@/lib/forms/useZodForm";
import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { TabsContent } from "@/components/ui/tabs";
import { ChevronDown, Plus, Users } from "lucide-react";
import type { TFunction } from "i18next";
import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { Button as UIButton } from "@/components/ui/button";
import { teamsTableKeys } from "@/app/(dashboard)/hooks/teams/useTeams";
import { parseAsString, useQueryState } from "nuqs";
import { TeamsTable } from "./TeamsPage/TeamsTable";
import AccessGroupSelector from "./common_components/AccessGroupSelector";
import MetadataKeyValueFields, {
  metadataPairsSchema,
  metadataPairsToObject,
} from "./common_components/MetadataKeyValueFields";
import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import PassThroughRoutesSelector from "./common_components/PassThroughRoutesSelector";
import AgentSelector from "./agent_management/AgentSelector";
import ModelAliasManager from "./common_components/ModelAliasManager";
import PremiumLoggingSettings from "./common_components/PremiumLoggingSettings";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./common_components/RouterSettingsAccordion";
import { fetchAvailableModelsForTeamOrKey } from "./key_team_helpers/fetch_available_models_team_key";
import type { Team } from "./key_team_helpers/key_list";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import { toast } from "@/lib/toast";
import { extractProxyErrorMessage } from "@/lib/http/client";
import BudgetDurationDropdown, {
  getBudgetDurationLabel,
  NEVER_RESETS_BUDGET_DURATION,
} from "./common_components/budget_duration_dropdown";
import { Organization, getDefaultTeamSettings, getGuardrailsList, getPoliciesList, teamDeleteCall } from "./networking";
import NumericalInput from "./shared/numerical_input";
import { ModelMaxBudget, ModelMaxBudgetField } from "./key_team_helpers/ModelMaxBudgetEditor";
import { modelMaxBudgetUpdate } from "./key_team_helpers/modelMaxBudgetPayload";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import SearchToolSelector from "./search_tools/SearchToolSelector";
import SkillSelector from "./skills/SkillSelector";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface TeamProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  isViewOnly?: boolean;
  premiumUser?: boolean;
}

import DeleteResourceModal from "./common_components/DeleteResourceModal";
import { teamCreateCall } from "./networking";
import { normalizeTeamModelSelection } from "./team/teamModelAccess";
import { ModelSelect } from "./ModelSelect/ModelSelect";

const SUPPRESSED_BY_DESCRIPTION = "";

const numericInputSchema = z.union([z.string(), z.number()]).optional();

const buildTeamCreateFieldsSchema = (t: TFunction) =>
  z.object({
    team_alias: z.string().min(1, t("teams.page.create.errors.teamNameRequired")),
    organization_id: z.string().nullish(),
    models: z.array(z.string()).optional(),
    max_budget: numericInputSchema,
    budget_duration: z.string().nullish(),
    tpm_limit: numericInputSchema,
    rpm_limit: numericInputSchema,
    tpd_limit: numericInputSchema,
    metadata: metadataPairsSchema.optional(),
    team_id: z.string().optional(),
    team_member_budget: z.number().optional(),
    team_member_key_duration: z.string().optional(),
    team_member_rpm_limit: numericInputSchema,
    team_member_tpm_limit: numericInputSchema,
    secret_manager_settings: z.string().optional(),
    guardrails: z.array(z.string()).optional(),
    disable_global_guardrails: z.boolean().optional(),
    require_trace_id: z.boolean().optional(),
    policies: z.array(z.string()).optional(),
    access_group_ids: z.array(z.string()).optional(),
    allowed_vector_store_ids: z.array(z.string()).optional(),
    allowed_passthrough_routes: z.array(z.string()).optional(),
    allowed_mcp_servers_and_groups: z
      .object({
        servers: z.array(z.string()),
        accessGroups: z.array(z.string()),
        toolsets: z.array(z.string()).optional(),
      })
      .optional(),
    mcp_tool_permissions: z.record(z.string(), z.array(z.string())).optional(),
    allowed_agents_and_groups: z.object({ agents: z.array(z.string()), accessGroups: z.array(z.string()) }).optional(),
    object_permission_search_tools: z.array(z.string()).optional(),
    object_permission_skills: z.array(z.string()).optional(),
  });

type TeamCreateFormValues = z.infer<ReturnType<typeof buildTeamCreateFieldsSchema>>;

const EMPTY_TEAM_CREATE_VALUES: TeamCreateFormValues = {
  team_alias: "",
  organization_id: null,
  models: [],
  max_budget: undefined,
  budget_duration: undefined,
  tpm_limit: undefined,
  rpm_limit: undefined,
  tpd_limit: undefined,
  metadata: [],
  team_id: undefined,
  team_member_budget: undefined,
  team_member_key_duration: undefined,
  team_member_rpm_limit: undefined,
  team_member_tpm_limit: undefined,
  secret_manager_settings: undefined,
  guardrails: undefined,
  disable_global_guardrails: undefined,
  require_trace_id: undefined,
  policies: undefined,
  access_group_ids: undefined,
  allowed_vector_store_ids: undefined,
  allowed_passthrough_routes: undefined,
  allowed_mcp_servers_and_groups: undefined,
  mcp_tool_permissions: {},
  allowed_agents_and_groups: undefined,
  object_permission_search_tools: undefined,
  object_permission_skills: undefined,
};

const ADDITIONAL_SETTINGS_FIELDS = [
  "team_id",
  "team_member_budget",
  "team_member_key_duration",
  "team_member_rpm_limit",
  "team_member_tpm_limit",
  "secret_manager_settings",
  "guardrails",
  "disable_global_guardrails",
  "require_trace_id",
  "policies",
  "access_group_ids",
  "allowed_vector_store_ids",
  "allowed_passthrough_routes",
] as const;
const MCP_SETTINGS_FIELDS = ["allowed_mcp_servers_and_groups", "mcp_tool_permissions"] as const;
const AGENT_SETTINGS_FIELDS = ["allowed_agents_and_groups"] as const;
const SEARCH_TOOL_SETTINGS_FIELDS = ["object_permission_search_tools"] as const;
const SKILL_SETTINGS_FIELDS = ["object_permission_skills"] as const;

const isParsableJson = (value: string | undefined): boolean => {
  if (!value) {
    return true;
  }
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
};

const canCreateOrManageTeams = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): boolean => {
  // Admin role always has permission
  if (userRole === "Admin") {
    return true;
  }

  // Check if user is an org_admin in any organization
  if (organizations && userID) {
    return organizations.some((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return false;
};

const getAdminOrganizations = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): Organization[] => {
  // Global Admin can see all organizations
  if (userRole === "Admin") {
    return organizations || [];
  }

  // Org Admin can only see organizations they're an admin for
  if (organizations && userID) {
    return organizations.filter((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return [];
};

// @deprecated
const Teams: React.FC<TeamProps> = ({ accessToken, userID, userRole, isViewOnly = false, premiumUser = false }) => {
  const { t } = useTranslation();
  const { data: organizationsData } = useOrganizations();
  const organizations = organizationsData ?? null;
  const { data: teamMetadataSchemaFields = [], isLoading: isTeamMetadataSchemaLoading } = useTeamMetadataSchema();
  const queryClient = useQueryClient();
  const refreshTeams = () => queryClient.invalidateQueries({ queryKey: teamsTableKeys.all });
  const [currentOrg] = useState<Organization | null>(null);

  const isOrgAdmin = userRole !== "Admin";
  const [additionalSettingsOpen, setAdditionalSettingsOpen] = useState(false);
  const [mcpSettingsOpen, setMcpSettingsOpen] = useState(false);
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [searchToolSettingsOpen, setSearchToolSettingsOpen] = useState(false);
  const [skillSettingsOpen, setSkillSettingsOpen] = useState(false);

  const adminOrgs = useMemo(
    () => getAdminOrganizations(userRole, userID, organizations),
    [userRole, userID, organizations],
  );

  const teamCreateSchema = useMemo(
    () =>
      buildTeamCreateFieldsSchema(t).superRefine((values, ctx) => {
        if (isOrgAdmin && !values.organization_id) {
          ctx.addIssue({ code: "custom", message: SUPPRESSED_BY_DESCRIPTION, path: ["organization_id"] });
        }
        const organizationIsStillPickable =
          values.organization_id == null ||
          organizations == null ||
          adminOrgs.some((org) => org.organization_id === values.organization_id);
        if (!organizationIsStillPickable) {
          ctx.addIssue({
            code: "custom",
            message: t("teams.page.create.errors.organizationNotPickable"),
            path: ["organization_id"],
          });
        }
        if (additionalSettingsOpen && !isParsableJson(values.secret_manager_settings)) {
          ctx.addIssue({ code: "custom", message: SUPPRESSED_BY_DESCRIPTION, path: ["secret_manager_settings"] });
        }
      }),
    [isOrgAdmin, additionalSettingsOpen, adminOrgs, organizations, t],
  );

  const form = useZodForm(teamCreateSchema, { defaultValues: EMPTY_TEAM_CREATE_VALUES });
  const watchedOrganizationId = form.watch("organization_id");
  const watchedMcpSelection = form.watch("allowed_mcp_servers_and_groups");
  const watchedToolPermissions = form.watch("mcp_tool_permissions");

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useQueryState("team", parseAsString.withOptions({ history: "push" }));
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const canEditAsProxyAdmin = !isViewOnly && isProxyAdminRole(userRole || "");
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<Team | null>(null);
  const [isTeamDeleting, setIsTeamDeleting] = useState(false);
  // Add this state near the other useState declarations
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const canViewPolicies = useCan("viewPolicies");
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [modelMaxBudget, setModelMaxBudget] = useState<ModelMaxBudget>({});
  const [teamMemberModelMaxBudget, setTeamMemberModelMaxBudget] = useState<ModelMaxBudget>({});
  const [routerSettings, setRouterSettings] = useState<RouterSettingsAccordionValue | null>(null);
  const [routerSettingsKey, setRouterSettingsKey] = useState<number>(0);

  const { data: defaultTeamSettings } = useQuery({
    queryKey: ["defaultTeamSettings"],
    queryFn: () => getDefaultTeamSettings(accessToken as string),
    enabled: isTeamModalVisible && accessToken != null,
    retry: false,
    staleTime: 60_000,
  });
  const defaultBudgetDuration: string | undefined = defaultTeamSettings?.values?.budget_duration ?? undefined;
  const budgetDurationPlaceholder = defaultBudgetDuration
    ? t("teams.page.create.fields.budgetDuration.defaultPlaceholder", {
        label: getBudgetDurationLabel(defaultBudgetDuration),
        duration: defaultBudgetDuration,
      })
    : undefined;

  // Add this useEffect to fetch guardrails
  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    fetchGuardrails();
    if (canViewPolicies) fetchPolicies();
  }, [accessToken, canViewPolicies]);

  const openCreateTeamModal = () => {
    // Org admins must scope a team to an org, so with exactly one we preselect it.
    // Proxy admins can create org-less teams, so the field stays optional regardless of org count.
    if (isOrgAdmin && adminOrgs.length === 1) {
      form.setValue("organization_id", adminOrgs[0].organization_id);
    }
    setIsTeamModalVisible(true);
  };

  const selectCreateTeamOrganization = (
    next: string | null,
    currentOrganizationId: string | null,
    onChange: (organizationId: string | null) => void,
  ) => {
    const nextOrganizationId = next;
    if (nextOrganizationId === currentOrganizationId) return;
    onChange(nextOrganizationId);
    form.setValue("models", []);
  };

  const resetCreateForm = () => {
    form.reset(EMPTY_TEAM_CREATE_VALUES);
    setAdditionalSettingsOpen(false);
    setMcpSettingsOpen(false);
    setAgentSettingsOpen(false);
    setSearchToolSettingsOpen(false);
    setLoggingSettings([]);
    setModelAliases({});
    setModelMaxBudget({});
    setTeamMemberModelMaxBudget({});
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    resetCreateForm();
  };

  const handleDelete = async (team: Team) => {
    // Set the team to delete and open the confirmation modal
    setTeamToDelete(team);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (teamToDelete == null || accessToken == null) {
      return;
    }

    try {
      setIsTeamDeleting(true);
      await teamDeleteCall(accessToken, teamToDelete.team_id);
      await refreshTeams();
      toast.success(t("teams.page.toasts.deleted"));
    } catch (error) {
      toast.fromError(t("teams.page.toasts.deleteFailed", { error: String(error) }));
    } finally {
      setIsTeamDeleting(false);
      setIsDeleteModalOpen(false);
      setTeamToDelete(null);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }
        const models = await fetchAvailableModelsForTeamOrKey(userID, userRole, accessToken);
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null) {
        let organizationId = formValues?.organization_id || currentOrg?.organization_id;
        if (organizationId === "" || typeof organizationId !== "string") {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }

        if (formValues.budget_duration === NEVER_RESETS_BUDGET_DURATION) {
          formValues.budget_duration = null;
        }

        toast.info(t("teams.page.toasts.creating"));

        const metadataObject = {
          ...metadataPairsToObject(formValues.metadata),
          ...(loggingSettings.length > 0 ? { logging: loggingSettings.filter((config) => config.callback_name) } : {}),
        };
        formValues.metadata = Object.keys(metadataObject).length > 0 ? JSON.stringify(metadataObject) : undefined;

        if (formValues.secret_manager_settings) {
          if (typeof formValues.secret_manager_settings === "string") {
            if (formValues.secret_manager_settings.trim() === "") {
              delete formValues.secret_manager_settings;
            } else {
              try {
                formValues.secret_manager_settings = JSON.parse(formValues.secret_manager_settings);
              } catch (e) {
                throw new Error("Failed to parse secret manager settings: " + e);
              }
            }
          }
        }

        const hasSearchTools =
          Array.isArray(formValues.object_permission_search_tools) &&
          formValues.object_permission_search_tools.length > 0;

        if (
          (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) ||
          (formValues.allowed_mcp_servers_and_groups &&
            (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.toolsets?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.toolPermissions))
        ) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
            formValues.object_permission.vector_stores = formValues.allowed_vector_store_ids;
            delete formValues.allowed_vector_store_ids;
          }
          if (formValues.allowed_mcp_servers_and_groups) {
            const { servers, accessGroups, toolsets } = formValues.allowed_mcp_servers_and_groups;
            if (servers && servers.length > 0) {
              formValues.object_permission.mcp_servers = servers;
            }
            if (accessGroups && accessGroups.length > 0) {
              formValues.object_permission.mcp_access_groups = accessGroups;
            }
            if (toolsets && toolsets.length > 0) {
              formValues.object_permission.mcp_toolsets = toolsets;
            }
            delete formValues.allowed_mcp_servers_and_groups;
          }

          if (formValues.mcp_tool_permissions && Object.keys(formValues.mcp_tool_permissions).length > 0) {
            formValues.object_permission.mcp_tool_permissions = formValues.mcp_tool_permissions;
            delete formValues.mcp_tool_permissions;
          }
        }

        // Transform allowed_mcp_access_groups into object_permission
        if (formValues.allowed_mcp_access_groups && formValues.allowed_mcp_access_groups.length > 0) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.mcp_access_groups = formValues.allowed_mcp_access_groups;
          delete formValues.allowed_mcp_access_groups;
        }

        // Handle agent permissions
        if (formValues.allowed_agents_and_groups) {
          const { agents, accessGroups } = formValues.allowed_agents_and_groups;
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (agents && agents.length > 0) {
            formValues.object_permission.agents = agents;
          }
          if (accessGroups && accessGroups.length > 0) {
            formValues.object_permission.agent_access_groups = accessGroups;
          }
          delete formValues.allowed_agents_and_groups;
        }

        if (hasSearchTools) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.search_tools = formValues.object_permission_search_tools;
          delete formValues.object_permission_search_tools;
        }

        if (Array.isArray(formValues.object_permission_skills) && formValues.object_permission_skills.length > 0) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.skills = formValues.object_permission_skills;
        }
        delete formValues.object_permission_skills;

        // Add model_aliases if any are defined
        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
        }

        if (Object.keys(modelMaxBudget).length > 0) {
          formValues.model_max_budget = modelMaxBudget;
        }

        // Add router_settings if any are defined
        if (routerSettings?.router_settings) {
          // Only include router_settings if it has at least one non-null value
          const hasValues = Object.values(routerSettings.router_settings).some(
            (value) => value !== null && value !== undefined && value !== "",
          );
          if (hasValues) {
            formValues.router_settings = routerSettings.router_settings;
          }
        }

        const { require_trace_id: requireTraceId, ...teamCreateValues } = formValues;
        const teamMemberModelBudgets = modelMaxBudgetUpdate(teamMemberModelMaxBudget, {});
        await teamCreateCall(accessToken, {
          ...teamCreateValues,
          ...(requireTraceId === undefined ? {} : { require_trace_id: requireTraceId }),
          ...(teamMemberModelBudgets !== undefined ? { team_member_model_max_budget: teamMemberModelBudgets } : {}),
          models: normalizeTeamModelSelection(formValues.models),
        });
        toast.success(t("teams.page.toasts.created"));
        await refreshTeams();
        resetCreateForm();
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      toast.fromError(t("teams.page.toasts.createFailed", { error: extractProxyErrorMessage(error) }));
    }
  };

  const mountedCreateValues = (values: TeamCreateFormValues): Record<string, unknown> => {
    const unmounted = new Set<string>([
      ...(additionalSettingsOpen ? [] : ADDITIONAL_SETTINGS_FIELDS),
      ...(additionalSettingsOpen && canViewPolicies ? [] : ["policies"]),
      ...(mcpSettingsOpen ? [] : MCP_SETTINGS_FIELDS),
      ...(agentSettingsOpen ? [] : AGENT_SETTINGS_FIELDS),
      ...(searchToolSettingsOpen ? [] : SEARCH_TOOL_SETTINGS_FIELDS),
      ...(skillSettingsOpen ? [] : SKILL_SETTINGS_FIELDS),
    ]);
    return Object.fromEntries(Object.entries(values).filter(([key]) => !unmounted.has(key)));
  };

  const onCreateSubmit = (values: TeamCreateFormValues) => handleCreate(mountedCreateValues(values));

  const is_team_admin = (team: any) => {
    if (team == null || team.members_with_roles == null) {
      return false;
    }
    for (let i = 0; i < team.members_with_roles.length; i++) {
      let member = team.members_with_roles[i];
      if (member.user_id == userID && member.role == "admin") {
        return true;
      }
    }
    return false;
  };

  const tabItems = [
    {
      key: "your-teams",
      label: t("teams.page.tabs.yourTeams"),
      className: "flex min-h-0 flex-1 flex-col",
      children: (
        <>
          <TeamsTable
            userRole={userRole}
            userID={userID}
            onSelectTeam={(team) => {
              setSelectedTeam(team);
              void setSelectedTeamId(team.team_id);
              setEditTeam(false);
            }}
            onEditTeam={(team) => {
              setSelectedTeam(team);
              void setSelectedTeamId(team.team_id);
              setEditTeam(true);
            }}
            onDeleteTeam={handleDelete}
          />

          <DeleteResourceModal
            isOpen={isDeleteModalOpen}
            title={t("teams.page.deleteModal.title")}
            alertMessage={(() => {
              const deleteKeyCount = teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0;
              return deleteKeyCount === 0
                ? undefined
                : t("teams.page.deleteModal.keysWarning", { count: deleteKeyCount });
            })()}
            message={t("teams.page.deleteModal.message")}
            resourceInformationTitle={t("teams.page.deleteModal.informationTitle")}
            resourceInformation={[
              { label: t("teams.page.deleteModal.teamId"), value: teamToDelete?.team_id, code: true },
              { label: t("teams.page.deleteModal.teamName"), value: teamToDelete?.team_alias },
              {
                label: t("teams.page.deleteModal.keys"),
                value: teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0,
              },
              { label: t("teams.page.deleteModal.members"), value: teamToDelete?.members_with_roles?.length },
            ]}
            requiredConfirmation={teamToDelete?.team_alias}
            onCancel={cancelDelete}
            onOk={confirmDelete}
            confirmLoading={isTeamDeleting}
          />
        </>
      ),
    },
    {
      key: "available-teams",
      label: t("teams.page.tabs.availableTeams"),
      className: "min-h-0 flex-1 overflow-y-auto",
      children: <AvailableTeamsPanel accessToken={accessToken} userID={userID} />,
    },
    ...(isProxyAdminRole(userRole || "")
      ? [
          {
            key: "default-settings",
            label: t("teams.page.tabs.defaultSettings"),
            className: "min-h-0 flex-1 overflow-y-auto",
            children: <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />,
          },
        ]
      : []),
  ];

  return (
    <Page className={selectedTeamId ? undefined : "h-full"}>
      {selectedTeamId ? (
        <TeamInfoView
          teamId={selectedTeamId}
          onUpdate={() => {
            refreshTeams();
          }}
          onClose={() => {
            setSelectedTeam(null);
            void setSelectedTeamId(null);
            setEditTeam(false);
          }}
          accessToken={accessToken}
          is_team_admin={is_team_admin(selectedTeam?.team_id === selectedTeamId ? selectedTeam : null)}
          is_proxy_admin={userRole == "Admin"}
          userModels={userModels}
          editTeam={editTeam}
          premiumUser={premiumUser}
        />
      ) : (
        <PageTabs defaultValue={tabItems[0].key}>
          <PageHeader>
            <PageHeaderTitle>
              <Users />
              {t("teams.page.header.title")}
            </PageHeaderTitle>
            <PageHeaderDescription>{t("teams.page.header.description")}</PageHeaderDescription>
            <PageHeaderControls>
              <PageTabsList>
                {canCreateOrManageTeams(userRole, userID, organizations) && (
                  <>
                    <UIButton onClick={openCreateTeamModal} data-testid="create-team-button">
                      <Plus className="size-4" />
                      {t("teams.page.header.createTeam")}
                    </UIButton>
                    <ToolbarSeparator className="mx-0 h-6" />
                  </>
                )}
                {tabItems.map((item) => (
                  <PageTabsTrigger key={item.key} value={item.key}>
                    {item.label}
                  </PageTabsTrigger>
                ))}
              </PageTabsList>
            </PageHeaderControls>
          </PageHeader>
          {tabItems.map((item) => (
            <TabsContent key={item.key} value={item.key} className={item.className}>
              {item.children}
            </TabsContent>
          ))}
        </PageTabs>
      )}

      {canCreateOrManageTeams(userRole, userID, organizations) && (
        <Dialog open={isTeamModalVisible} onOpenChange={(open) => !open && handleCancel()}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
            <DialogHeader>
              <DialogTitle>{t("teams.page.create.title")}</DialogTitle>
            </DialogHeader>
            <TooltipProvider>
              <form onSubmit={form.handleSubmit(onCreateSubmit)}>
                <FieldGroup>
                  <FormField control={form.control} name="team_alias" label={t("teams.page.create.fields.teamName")}>
                    {({ ref, value, ...field }) => (
                      <UIInput {...field} ref={ref} value={value ?? ""} data-testid="team-name-input" />
                    )}
                  </FormField>
                  {(() => {
                    const isSingleOrg = adminOrgs.length === 1;
                    const hasNoOrgs = adminOrgs.length === 0;
                    const soleOrganizationId = isSingleOrg ? adminOrgs[0].organization_id ?? null : null;

                    return (
                      <>
                        <FormField
                          control={form.control}
                          name="organization_id"
                          className="mt-8"
                          label={labelWithDocsHint(
                            t("teams.page.create.fields.organization.label"),
                            t("teams.page.create.fields.organization.hint"),
                            "https://docs.litellm.ai/docs/proxy/user_management_heirarchy",
                          )}
                          description={
                            isOrgAdmin && isSingleOrg
                              ? t("teams.page.create.fields.organization.singleOrgDescription")
                              : isOrgAdmin
                                ? t("teams.page.create.fields.organization.requiredDescription")
                                : undefined
                          }
                        >
                          {({ id, value, onChange }) => (
                            <SearchSelect
                              inputId={id}
                              value={value ?? ""}
                              options={adminOrgs.map((org) => ({
                                value: org.organization_id ?? "",
                                label: org.organization_alias ?? "",
                                sublabel: org.organization_id ?? "",
                              }))}
                              disabled={isOrgAdmin && soleOrganizationId !== null && value === soleOrganizationId}
                              allowClear={!isOrgAdmin}
                              placeholder={
                                hasNoOrgs
                                  ? t("teams.page.create.fields.organization.noOrganizations")
                                  : t("teams.page.create.fields.organization.placeholder")
                              }
                              emptyText={t("teams.page.create.fields.organization.noOrganizations")}
                              onValueChange={(next) => selectCreateTeamOrganization(next, value ?? null, onChange)}
                            />
                          )}
                        </FormField>

                        {isOrgAdmin && !isSingleOrg && adminOrgs.length > 1 && (
                          <div className="mb-8 rounded-md border border-info/20 bg-info/10 p-4">
                            <span className="text-sm text-info">
                              {t("teams.page.create.fields.organization.selectPrompt")}
                            </span>
                          </div>
                        )}
                      </>
                    );
                  })()}
                  <FormField
                    control={form.control}
                    name="models"
                    label={labelWithHint(
                      t("teams.page.create.fields.models.label"),
                      t("teams.page.create.fields.models.hint"),
                    )}
                  >
                    {({ id, value, onChange }) => (
                      <ModelSelect
                        id={id}
                        value={value ?? []}
                        onChange={onChange}
                        organizationID={watchedOrganizationId ?? undefined}
                        options={{
                          includeSpecialOptions: true,
                          showAllProxyModelsOverride: !watchedOrganizationId,
                        }}
                        context="team"
                        dataTestId="create-team-models-select"
                      />
                    )}
                  </FormField>

                  <FormField control={form.control} name="max_budget" label={t("teams.page.create.fields.maxBudget")}>
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={0.01} precision={2} width={200} />
                    )}
                  </FormField>
                  <FormField
                    control={form.control}
                    name="budget_duration"
                    className="mt-8"
                    label={t("teams.page.create.fields.budgetDuration.label")}
                  >
                    {({ id, value, onChange }) => (
                      <BudgetDurationDropdown
                        id={id}
                        showNeverResets={Boolean(defaultBudgetDuration)}
                        placeholder={budgetDurationPlaceholder}
                        value={value}
                        onChange={(next) => onChange(next ?? undefined)}
                      />
                    )}
                  </FormField>
                  <ModelMaxBudgetField
                    key={`model-max-budget-${routerSettingsKey}`}
                    premiumUser={premiumUser}
                    value={modelMaxBudget}
                    onChange={setModelMaxBudget}
                    availableModels={userModels}
                    hint={t("teams.page.create.fields.modelMaxBudgetHint")}
                  />
                  <FormField control={form.control} name="tpm_limit" label={t("teams.page.create.fields.tpmLimit")}>
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                    )}
                  </FormField>
                  <FormField control={form.control} name="rpm_limit" label={t("teams.page.create.fields.rpmLimit")}>
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                    )}
                  </FormField>
                  <FormField
                    control={form.control}
                    name="tpd_limit"
                    label={labelWithHint(
                      t("teams.page.create.fields.tpdLimit.label"),
                      t("teams.page.create.fields.tpdLimit.hint"),
                    )}
                  >
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                    )}
                  </FormField>
                  <Field>
                    <FieldLabel>{t("teams.page.create.fields.metadata.label")}</FieldLabel>
                    <MetadataKeyValueFields
                      control={form.control}
                      getValues={form.getValues}
                      name="metadata"
                      schemaFields={teamMetadataSchemaFields}
                      schemaLoading={isTeamMetadataSchemaLoading}
                    />
                    <FieldDescription>
                      {t("teams.page.create.fields.metadata.description", { example: '{"region": "us"}' })}
                    </FieldDescription>
                  </Field>

                  <Collapsible
                    open={additionalSettingsOpen}
                    onOpenChange={setAdditionalSettingsOpen}
                    className="mt-20 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.additional")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FieldGroup>
                        <FormField
                          control={form.control}
                          name="team_id"
                          label={t("teams.page.create.fields.teamId.label")}
                          description={t("teams.page.create.fields.teamId.description")}
                        >
                          {({ ref, value, ...field }) => <UIInput {...field} ref={ref} value={value ?? ""} />}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="team_member_budget"
                          label={labelWithHint(
                            t("teams.page.create.fields.teamMemberBudget.label"),
                            t("teams.page.create.fields.teamMemberBudget.hint"),
                          )}
                        >
                          {({ ref, value, onChange, ...field }) => (
                            <NumericalInput
                              {...field}
                              ref={ref}
                              value={value ?? ""}
                              onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                                onChange(event.target.value ? Number(event.target.value) : undefined)
                              }
                              step={0.01}
                              precision={2}
                              width={200}
                            />
                          )}
                        </FormField>
                        <ModelMaxBudgetField
                          key={`team-member-model-max-budget-${routerSettingsKey}`}
                          label={t("teams.page.create.fields.teamMemberModelMaxBudget.label")}
                          premiumUser={premiumUser}
                          value={teamMemberModelMaxBudget}
                          onChange={setTeamMemberModelMaxBudget}
                          availableModels={userModels}
                          hint={t("teams.page.create.fields.teamMemberModelMaxBudget.hint")}
                        />
                        <FormField
                          control={form.control}
                          name="team_member_key_duration"
                          label={labelWithHint(
                            t("teams.page.create.fields.teamMemberKeyDuration.label"),
                            t("teams.page.create.fields.teamMemberKeyDuration.hint"),
                          )}
                        >
                          {({ ref, value, ...field }) => (
                            <UIInput
                              {...field}
                              ref={ref}
                              value={value ?? ""}
                              placeholder={t("teams.page.create.fields.teamMemberKeyDuration.placeholder")}
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="team_member_rpm_limit"
                          label={labelWithHint(
                            t("teams.page.create.fields.teamMemberRpmLimit.label"),
                            t("teams.page.create.fields.teamMemberRpmLimit.hint"),
                          )}
                        >
                          {({ ref, value, ...field }) => (
                            <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="team_member_tpm_limit"
                          label={labelWithHint(
                            t("teams.page.create.fields.teamMemberTpmLimit.label"),
                            t("teams.page.create.fields.teamMemberTpmLimit.hint"),
                          )}
                        >
                          {({ ref, value, ...field }) => (
                            <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="secret_manager_settings"
                          label={t("teams.page.create.fields.secretManagerSettings.label")}
                          description={
                            premiumUser
                              ? t("teams.page.create.fields.secretManagerSettings.premiumDescription")
                              : t("teams.page.create.fields.secretManagerSettings.upgradeDescription")
                          }
                        >
                          {({ ref, value, ...field }) => (
                            <Textarea
                              {...field}
                              ref={ref}
                              value={value ?? ""}
                              rows={4}
                              placeholder='{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
                              disabled={!premiumUser}
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="guardrails"
                          className="mt-8"
                          label={labelWithDocsHint(
                            t("teams.page.create.fields.guardrails.label"),
                            t("teams.page.create.fields.guardrails.hint"),
                            "https://docs.litellm.ai/docs/proxy/guardrails/quick_start",
                          )}
                          description={t("teams.page.create.fields.guardrails.description")}
                        >
                          {({ id, value, onChange }) => (
                            <TagsInput
                              id={id}
                              value={value ?? []}
                              onValueChange={onChange}
                              options={guardrailsList.map((name) => ({ value: name, label: name }))}
                              placeholder={t("teams.page.create.fields.guardrails.placeholder")}
                            />
                          )}
                        </FormField>
                        {isProxyAdminRole(userRole || "") && (
                          <FormField
                            control={form.control}
                            name="disable_global_guardrails"
                            className="mt-4"
                            label={labelWithHint(
                              t("teams.page.create.fields.disableGlobalGuardrails.label"),
                              t("teams.page.create.fields.disableGlobalGuardrails.hint"),
                            )}
                            description={
                              premiumUser
                                ? t("teams.page.create.fields.disableGlobalGuardrails.premiumDescription")
                                : t("teams.page.create.fields.disableGlobalGuardrails.upgradeDescription")
                            }
                          >
                            {({ id, value, onChange }) => (
                              <Switch
                                id={id}
                                disabled={!premiumUser}
                                checked={value === true}
                                onCheckedChange={onChange}
                              />
                            )}
                          </FormField>
                        )}
                        {canEditAsProxyAdmin && (
                          <FormField
                            control={form.control}
                            name="require_trace_id"
                            className="mt-4"
                            label={labelWithHint(
                              t("teams.page.create.fields.requireTraceId.label"),
                              t("teams.page.create.fields.requireTraceId.hint"),
                            )}
                          >
                            {({ id, value, onChange }) => (
                              <Switch id={id} checked={value === true} onCheckedChange={onChange} />
                            )}
                          </FormField>
                        )}
                        {canViewPolicies && (
                          <FormField
                            control={form.control}
                            name="policies"
                            className="mt-8"
                            label={labelWithDocsHint(
                              t("teams.page.create.fields.policies.label"),
                              t("teams.page.create.fields.policies.hint"),
                              "https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies",
                            )}
                            description={t("teams.page.create.fields.policies.description")}
                          >
                            {({ id, value, onChange }) => (
                              <TagsInput
                                id={id}
                                value={value ?? []}
                                onValueChange={onChange}
                                options={policiesList.map((name) => ({ value: name, label: name }))}
                                placeholder={t("teams.page.create.fields.policies.placeholder")}
                              />
                            )}
                          </FormField>
                        )}
                        <FormField
                          control={form.control}
                          name="access_group_ids"
                          className="mt-8"
                          label={labelWithHint(
                            t("teams.page.create.fields.accessGroups.label"),
                            t("teams.page.create.fields.accessGroups.hint"),
                          )}
                          description={t("teams.page.create.fields.accessGroups.description")}
                        >
                          {({ value, onChange }) => (
                            <AccessGroupSelector
                              value={value}
                              onChange={onChange}
                              placeholder={t("teams.page.create.fields.accessGroups.placeholder")}
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="allowed_vector_store_ids"
                          className="mt-8"
                          label={labelWithHint(
                            t("teams.page.create.fields.vectorStores.label"),
                            t("teams.page.create.fields.vectorStores.hint"),
                          )}
                          description={t("teams.page.create.fields.vectorStores.description")}
                        >
                          {({ value, onChange }) => (
                            <VectorStoreSelector
                              onChange={onChange}
                              value={value}
                              accessToken={accessToken || ""}
                              placeholder={t("teams.page.create.fields.vectorStores.placeholder")}
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="allowed_passthrough_routes"
                          className="mt-8"
                          label={
                            !premiumUser
                              ? labelWithHint(
                                  t("teams.page.create.fields.passThroughRoutes.label"),
                                  t("teams.page.create.fields.passThroughRoutes.upgradeHint"),
                                )
                              : !isProxyAdminRole(userRole || "")
                                ? labelWithHint(
                                    t("teams.page.create.fields.passThroughRoutes.label"),
                                    t("teams.page.create.fields.passThroughRoutes.proxyAdminOnlyHint"),
                                  )
                                : t("teams.page.create.fields.passThroughRoutes.label")
                          }
                        >
                          {({ value, onChange }) => (
                            <PassThroughRoutesSelector
                              value={value}
                              onChange={onChange}
                              accessToken={accessToken || ""}
                              placeholder={t("teams.page.create.fields.passThroughRoutes.placeholder")}
                              disabled={!premiumUser || !isProxyAdminRole(userRole || "")}
                            />
                          )}
                        </FormField>
                      </FieldGroup>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible
                    open={mcpSettingsOpen}
                    onOpenChange={setMcpSettingsOpen}
                    className="mt-8 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.mcp")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="allowed_mcp_servers_and_groups"
                        className="mt-4"
                        label={labelWithHint(
                          t("teams.page.create.fields.mcpServers.label"),
                          t("teams.page.create.fields.mcpServers.hint"),
                        )}
                        description={t("teams.page.create.fields.mcpServers.description")}
                      >
                        {({ value, onChange }) => (
                          <MCPServerSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder={t("teams.page.create.fields.mcpServers.placeholder")}
                            allowAllProxyMcpServers={isProxyAdminRole(userRole || "")}
                          />
                        )}
                      </FormField>

                      <div className="mt-6">
                        <MCPToolPermissions
                          accessToken={accessToken || ""}
                          selectedServers={watchedMcpSelection?.servers || []}
                          selectedAccessGroups={watchedMcpSelection?.accessGroups || []}
                          selectedToolsets={watchedMcpSelection?.toolsets || []}
                          toolPermissions={watchedToolPermissions || {}}
                          onChange={(toolPerms) => form.setValue("mcp_tool_permissions", toolPerms)}
                        />
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible
                    open={agentSettingsOpen}
                    onOpenChange={setAgentSettingsOpen}
                    className="mt-8 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.agent")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="allowed_agents_and_groups"
                        className="mt-4"
                        label={labelWithHint(
                          t("teams.page.create.fields.agents.label"),
                          t("teams.page.create.fields.agents.hint"),
                        )}
                        description={t("teams.page.create.fields.agents.description")}
                      >
                        {({ value, onChange }) => (
                          <AgentSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder={t("teams.page.create.fields.agents.placeholder")}
                          />
                        )}
                      </FormField>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible
                    open={searchToolSettingsOpen}
                    onOpenChange={setSearchToolSettingsOpen}
                    className="mt-8 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.searchTool")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="object_permission_search_tools"
                        className="mt-4"
                        label={labelWithHint(
                          t("teams.page.create.fields.searchTools.label"),
                          t("teams.page.create.fields.searchTools.hint"),
                        )}
                        description={t("teams.page.create.fields.searchTools.description")}
                      >
                        {({ value, onChange }) => (
                          <SearchToolSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder={t("teams.page.create.fields.searchTools.placeholder")}
                          />
                        )}
                      </FormField>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible
                    open={skillSettingsOpen}
                    onOpenChange={setSkillSettingsOpen}
                    className="mt-8 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.skill")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="object_permission_skills"
                        className="mt-4"
                        label={labelWithHint(
                          t("teams.page.create.fields.skills.label"),
                          t("teams.page.create.fields.skills.hint"),
                        )}
                        description={t("teams.page.create.fields.skills.description")}
                      >
                        {({ value, onChange }) => (
                          <SkillSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder={t("teams.page.create.fields.skills.placeholder")}
                          />
                        )}
                      </FormField>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible className="mt-8 mb-8 overflow-hidden rounded-lg border">
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.logging")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <div className="mt-4">
                        <PremiumLoggingSettings
                          value={loggingSettings}
                          onChange={setLoggingSettings}
                          premiumUser={premiumUser}
                        />
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible
                    key={`router-settings-accordion-${routerSettingsKey}`}
                    className="mt-8 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.router")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <div className="mt-4 w-full">
                        <RouterSettingsAccordion
                          key={routerSettingsKey}
                          accessToken={accessToken || ""}
                          value={routerSettings || undefined}
                          onChange={setRouterSettings}
                          modelData={
                            userModels.length > 0
                              ? { data: userModels.map((model) => ({ model_name: model })) }
                              : undefined
                          }
                        />
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible className="mt-8 mb-8 overflow-hidden rounded-lg border">
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>{t("teams.page.create.sections.modelAliases")}</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <div className="mt-4">
                        <p className="mb-4 block text-sm text-muted-foreground">
                          {t("teams.page.create.fields.modelAliasesDescription")}
                        </p>
                        <ModelAliasManager
                          accessToken={accessToken || ""}
                          initialModelAliases={modelAliases}
                          onAliasUpdate={setModelAliases}
                          showExampleConfig={false}
                        />
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </FieldGroup>
                <div className="mt-[10px] text-right">
                  <UIButton type="submit" data-testid="create-team-submit">
                    {t("teams.page.create.submit")}
                  </UIButton>
                </div>
              </form>
            </TooltipProvider>
          </DialogContent>
        </Dialog>
      )}
    </Page>
  );
};

export default Teams;
