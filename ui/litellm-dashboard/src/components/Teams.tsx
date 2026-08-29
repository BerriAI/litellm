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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChevronDown, Plus, Users } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { z } from "zod/v4";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/shared/PageHeader";
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
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import SearchToolSelector from "./search_tools/SearchToolSelector";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface TeamProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  premiumUser?: boolean;
}

import DeleteResourceModal from "./common_components/DeleteResourceModal";
import { teamCreateCall } from "./networking";
import { normalizeTeamModelSelection } from "./team/teamModelAccess";
import { ModelSelect } from "./ModelSelect/ModelSelect";

const SUPPRESSED_BY_DESCRIPTION = "";

const numericInputSchema = z.union([z.string(), z.number()]).optional();

const teamCreateFieldsSchema = z.object({
  team_alias: z.string().min(1, "Please input a team name"),
  organization_id: z.string().nullish(),
  models: z.array(z.string()).optional(),
  max_budget: numericInputSchema,
  budget_duration: z.string().nullish(),
  tpm_limit: numericInputSchema,
  rpm_limit: numericInputSchema,
  metadata: metadataPairsSchema.optional(),
  team_id: z.string().optional(),
  team_member_budget: z.number().optional(),
  team_member_key_duration: z.string().optional(),
  team_member_rpm_limit: numericInputSchema,
  team_member_tpm_limit: numericInputSchema,
  secret_manager_settings: z.string().optional(),
  guardrails: z.array(z.string()).optional(),
  disable_global_guardrails: z.boolean().optional(),
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
});

type TeamCreateFormValues = z.infer<typeof teamCreateFieldsSchema>;

const EMPTY_TEAM_CREATE_VALUES: TeamCreateFormValues = {
  team_alias: "",
  organization_id: null,
  models: [],
  max_budget: undefined,
  budget_duration: undefined,
  tpm_limit: undefined,
  rpm_limit: undefined,
  metadata: [],
  team_id: undefined,
  team_member_budget: undefined,
  team_member_key_duration: undefined,
  team_member_rpm_limit: undefined,
  team_member_tpm_limit: undefined,
  secret_manager_settings: undefined,
  guardrails: undefined,
  disable_global_guardrails: undefined,
  policies: undefined,
  access_group_ids: undefined,
  allowed_vector_store_ids: undefined,
  allowed_passthrough_routes: undefined,
  allowed_mcp_servers_and_groups: undefined,
  mcp_tool_permissions: {},
  allowed_agents_and_groups: undefined,
  object_permission_search_tools: undefined,
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
  "policies",
  "access_group_ids",
  "allowed_vector_store_ids",
  "allowed_passthrough_routes",
] as const;
const MCP_SETTINGS_FIELDS = ["allowed_mcp_servers_and_groups", "mcp_tool_permissions"] as const;
const AGENT_SETTINGS_FIELDS = ["allowed_agents_and_groups"] as const;
const SEARCH_TOOL_SETTINGS_FIELDS = ["object_permission_search_tools"] as const;

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
const Teams: React.FC<TeamProps> = ({ accessToken, userID, userRole, premiumUser = false }) => {
  const { data: organizationsData } = useOrganizations();
  const organizations = organizationsData ?? null;
  const { data: teamMetadataSchemaFields = [], isLoading: isTeamMetadataSchemaLoading } = useTeamMetadataSchema();
  const queryClient = useQueryClient();
  const refreshTeams = () => queryClient.invalidateQueries({ queryKey: teamsTableKeys.all });
  const [currentOrg] = useState<Organization | null>(null);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] = useState<Organization | null>(null);

  const isOrgAdmin = userRole !== "Admin";
  const [additionalSettingsOpen, setAdditionalSettingsOpen] = useState(false);
  const [mcpSettingsOpen, setMcpSettingsOpen] = useState(false);
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [searchToolSettingsOpen, setSearchToolSettingsOpen] = useState(false);

  const teamCreateSchema = useMemo(
    () =>
      teamCreateFieldsSchema.superRefine((values, ctx) => {
        if (isOrgAdmin && !values.organization_id) {
          ctx.addIssue({ code: "custom", message: SUPPRESSED_BY_DESCRIPTION, path: ["organization_id"] });
        }
        if (additionalSettingsOpen && !isParsableJson(values.secret_manager_settings)) {
          ctx.addIssue({ code: "custom", message: SUPPRESSED_BY_DESCRIPTION, path: ["secret_manager_settings"] });
        }
      }),
    [isOrgAdmin, additionalSettingsOpen],
  );

  const form = useZodForm(teamCreateSchema, { defaultValues: EMPTY_TEAM_CREATE_VALUES });
  const watchedOrganizationId = form.watch("organization_id");
  const watchedMcpSelection = form.watch("allowed_mcp_servers_and_groups");
  const watchedToolPermissions = form.watch("mcp_tool_permissions");

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useQueryState("team", parseAsString.withOptions({ history: "push" }));
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
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
    ? `Default: ${getBudgetDurationLabel(defaultBudgetDuration)} (${defaultBudgetDuration})`
    : "n/a";

  useEffect(() => {
    form.setValue("models", []);
  }, [currentOrgForCreateTeam, userModels]);

  // Handle organization preselection when modal opens
  useEffect(() => {
    if (isTeamModalVisible) {
      const adminOrgs = getAdminOrganizations(userRole, userID, organizations);

      // Org admins must scope a team to an org, so with exactly one we preselect it.
      // Proxy admins can create org-less teams, so the field stays optional regardless of org count.
      if (isOrgAdmin && adminOrgs.length === 1) {
        const org = adminOrgs[0];
        form.setValue("organization_id", org.organization_id);
        setCurrentOrgForCreateTeam(org);
      } else {
        form.setValue("organization_id", currentOrg?.organization_id || null);
        setCurrentOrgForCreateTeam(currentOrg);
      }
    }
  }, [isTeamModalVisible, isOrgAdmin, userRole, userID, organizations, currentOrg]);

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

  const resetCreateForm = () => {
    form.reset(EMPTY_TEAM_CREATE_VALUES);
    setAdditionalSettingsOpen(false);
    setMcpSettingsOpen(false);
    setAgentSettingsOpen(false);
    setSearchToolSettingsOpen(false);
    setLoggingSettings([]);
    setModelAliases({});
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
      toast.success("Team deleted successfully");
    } catch (error) {
      toast.fromError("Error deleting the team: " + error);
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

        toast.info("Creating Team");

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
            const { servers, accessGroups } = formValues.allowed_mcp_servers_and_groups;
            if (servers && servers.length > 0) {
              formValues.object_permission.mcp_servers = servers;
            }
            if (accessGroups && accessGroups.length > 0) {
              formValues.object_permission.mcp_access_groups = accessGroups;
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

        // Add model_aliases if any are defined
        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
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

        await teamCreateCall(accessToken, { ...formValues, models: normalizeTeamModelSelection(formValues.models) });
        toast.success("Team created");
        await refreshTeams();
        resetCreateForm();
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      toast.fromError("Error creating the team: " + extractProxyErrorMessage(error));
    }
  };

  const mountedCreateValues = (values: TeamCreateFormValues): Record<string, unknown> => {
    const unmounted = new Set<string>([
      ...(additionalSettingsOpen ? [] : ADDITIONAL_SETTINGS_FIELDS),
      ...(additionalSettingsOpen && canViewPolicies ? [] : ["policies"]),
      ...(mcpSettingsOpen ? [] : MCP_SETTINGS_FIELDS),
      ...(agentSettingsOpen ? [] : AGENT_SETTINGS_FIELDS),
      ...(searchToolSettingsOpen ? [] : SEARCH_TOOL_SETTINGS_FIELDS),
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
      label: "Your Teams",
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
            title="Delete Team?"
            alertMessage={(() => {
              const deleteKeyCount = teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0;
              return deleteKeyCount === 0
                ? undefined
                : `Warning: This team has ${deleteKeyCount} keys associated with it. Deleting the team will also delete all associated keys, along with any models created for this team. This action is irreversible.`;
            })()}
            message="Are you sure you want to delete this team, all its keys, and any models created for it? This action cannot be undone."
            resourceInformationTitle="Team Information"
            resourceInformation={[
              { label: "Team ID", value: teamToDelete?.team_id, code: true },
              { label: "Team Name", value: teamToDelete?.team_alias },
              {
                label: "Keys",
                value: teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0,
              },
              { label: "Members", value: teamToDelete?.members_with_roles?.length },
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
      label: "Available Teams",
      children: <AvailableTeamsPanel accessToken={accessToken} userID={userID} />,
    },
    ...(isProxyAdminRole(userRole || "")
      ? [
          {
            key: "default-settings",
            label: "Default Team Settings",
            children: <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />,
          },
        ]
      : []),
  ];

  return (
    <main className={selectedTeamId ? "px-12 py-6" : "p-8"}>
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
        <Tabs defaultValue={tabItems[0].key} className="gap-6">
          <PageHeader
            icon={<Users />}
            title="Teams"
            subtitle="Manage teams, members, and their access to models and budgets"
            primaryAction={
              canCreateOrManageTeams(userRole, userID, organizations) ? (
                <UIButton onClick={() => setIsTeamModalVisible(true)} data-testid="create-team-button">
                  <Plus className="size-4" />
                  Create Team
                </UIButton>
              ) : undefined
            }
            tabs={({ leadingControls }) => (
              <TabsList
                variant="line"
                className="gap-0 p-0 [&>[data-slot=tabs-trigger]+[data-slot=tabs-trigger]]:ml-[22px]"
              >
                {leadingControls}
                {tabItems.map((item) => (
                  <TabsTrigger
                    key={item.key}
                    value={item.key}
                    className="flex-none px-0 py-[7px] data-active:font-semibold"
                  >
                    {item.label}
                  </TabsTrigger>
                ))}
              </TabsList>
            )}
          />
          {tabItems.map((item) => (
            <TabsContent key={item.key} value={item.key}>
              {item.children}
            </TabsContent>
          ))}
        </Tabs>
      )}

      {canCreateOrManageTeams(userRole, userID, organizations) && (
        <Dialog open={isTeamModalVisible} onOpenChange={(open) => !open && handleCancel()}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
            <DialogHeader>
              <DialogTitle>Create Team</DialogTitle>
            </DialogHeader>
            <TooltipProvider>
              <form onSubmit={form.handleSubmit(onCreateSubmit)}>
                <FieldGroup>
                  <FormField control={form.control} name="team_alias" label="Team Name">
                    {({ ref, value, ...field }) => (
                      <UIInput {...field} ref={ref} value={value ?? ""} data-testid="team-name-input" />
                    )}
                  </FormField>
                  {(() => {
                    const adminOrgs = getAdminOrganizations(userRole, userID, organizations);
                    const isSingleOrg = adminOrgs.length === 1;
                    const hasNoOrgs = adminOrgs.length === 0;

                    return (
                      <>
                        <FormField
                          control={form.control}
                          name="organization_id"
                          className="mt-8"
                          label={labelWithDocsHint(
                            "Organization",
                            "Organizations can have multiple teams. Learn more about the user management hierarchy",
                            "https://docs.litellm.ai/docs/proxy/user_management_heirarchy",
                          )}
                          description={
                            isOrgAdmin && isSingleOrg
                              ? "You can only create teams within this organization"
                              : isOrgAdmin
                                ? "required"
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
                              disabled={isOrgAdmin && isSingleOrg}
                              allowClear={!isOrgAdmin}
                              placeholder={
                                hasNoOrgs ? "No organizations available" : "Search or select an Organization"
                              }
                              emptyText="No organizations available"
                              onValueChange={(next) => {
                                onChange(next === "" ? null : next);
                                setCurrentOrgForCreateTeam(
                                  adminOrgs.find((org) => org.organization_id === next) ?? null,
                                );
                              }}
                            />
                          )}
                        </FormField>

                        {isOrgAdmin && !isSingleOrg && adminOrgs.length > 1 && (
                          <div className="mb-8 rounded-md border border-info/20 bg-info/10 p-4">
                            <span className="text-sm text-info">
                              Please select an organization to create a team for. You can only create teams within
                              organizations where you are an admin.
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
                      "Models",
                      "These are the models that your selected team has access to. Leave empty to grant no models directly, e.g. when the team gets its models from access groups",
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

                  <FormField control={form.control} name="max_budget" label="Max Budget (USD)">
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={0.01} precision={2} width={200} />
                    )}
                  </FormField>
                  <FormField control={form.control} name="budget_duration" className="mt-8" label="Reset Budget">
                    {({ id, value, onChange }) => (
                      <BudgetDurationDropdown
                        id={id}
                        showNeverResets
                        placeholder={budgetDurationPlaceholder}
                        value={value}
                        onChange={onChange}
                      />
                    )}
                  </FormField>
                  <FormField control={form.control} name="tpm_limit" label="Tokens per minute Limit (TPM)">
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                    )}
                  </FormField>
                  <FormField control={form.control} name="rpm_limit" label="Requests per minute Limit (RPM)">
                    {({ ref, value, ...field }) => (
                      <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                    )}
                  </FormField>
                  <Field>
                    <FieldLabel>Metadata</FieldLabel>
                    <MetadataKeyValueFields
                      control={form.control}
                      getValues={form.getValues}
                      name="metadata"
                      schemaFields={teamMetadataSchemaFields}
                      schemaLoading={isTeamMetadataSchemaLoading}
                    />
                    <FieldDescription>
                      Values are saved as text. Enter JSON for typed values, e.g. 3, true, or {'{"region": "us"}'}.
                    </FieldDescription>
                  </Field>

                  <Collapsible
                    open={additionalSettingsOpen}
                    onOpenChange={setAdditionalSettingsOpen}
                    className="mt-20 mb-8 overflow-hidden rounded-lg border"
                  >
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>Additional Settings</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FieldGroup>
                        <FormField
                          control={form.control}
                          name="team_id"
                          label="Team ID"
                          description="ID of the team you want to create. If not provided, it will be generated automatically."
                        >
                          {({ ref, value, ...field }) => <UIInput {...field} ref={ref} value={value ?? ""} />}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="team_member_budget"
                          label={labelWithHint(
                            "Team Member Budget (USD)",
                            "This is the individual budget for a user in the team.",
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
                        <FormField
                          control={form.control}
                          name="team_member_key_duration"
                          label={labelWithHint(
                            "Team Member Key Duration (eg: 1d, 1mo)",
                            "Set a limit to the duration of a team member's key. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days), 1mo (month)",
                          )}
                        >
                          {({ ref, value, ...field }) => (
                            <UIInput {...field} ref={ref} value={value ?? ""} placeholder="e.g., 30d" />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="team_member_rpm_limit"
                          label={labelWithHint(
                            "Team Member RPM Limit",
                            "The RPM (Requests Per Minute) limit for individual team members",
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
                            "Team Member TPM Limit",
                            "The TPM (Tokens Per Minute) limit for individual team members",
                          )}
                        >
                          {({ ref, value, ...field }) => (
                            <NumericalInput {...field} ref={ref} value={value ?? ""} step={1} width={400} />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="secret_manager_settings"
                          label="Secret Manager Settings"
                          description={
                            premiumUser
                              ? "Enter secret manager configuration as a JSON object."
                              : "Premium feature - Upgrade to manage secret manager settings."
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
                            "Guardrails",
                            "Setup your first guardrail",
                            "https://docs.litellm.ai/docs/proxy/guardrails/quick_start",
                          )}
                          description="Select existing guardrails or enter new ones"
                        >
                          {({ id, value, onChange }) => (
                            <TagsInput
                              id={id}
                              value={value ?? []}
                              onValueChange={onChange}
                              options={guardrailsList.map((name) => ({ value: name, label: name }))}
                              placeholder="Select or enter guardrails"
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="disable_global_guardrails"
                          className="mt-4"
                          label={labelWithHint(
                            "Disable Global Guardrails",
                            "When enabled, this team will bypass any guardrails configured to run on every request (global guardrails)",
                          )}
                          description={
                            premiumUser
                              ? "Bypass global guardrails for this team"
                              : "Premium feature - Upgrade to disable global guardrails by team"
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
                        {canViewPolicies && (
                          <FormField
                            control={form.control}
                            name="policies"
                            className="mt-8"
                            label={labelWithDocsHint(
                              "Policies",
                              "Apply policies to this team to control guardrails and other settings",
                              "https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies",
                            )}
                            description="Select existing policies or enter new ones"
                          >
                            {({ id, value, onChange }) => (
                              <TagsInput
                                id={id}
                                value={value ?? []}
                                onValueChange={onChange}
                                options={policiesList.map((name) => ({ value: name, label: name }))}
                                placeholder="Select or enter policies"
                              />
                            )}
                          </FormField>
                        )}
                        <FormField
                          control={form.control}
                          name="access_group_ids"
                          className="mt-8"
                          label={labelWithHint(
                            "Access Groups",
                            "Assign access groups to this team. Access groups control which models, MCP servers, and agents this team can use",
                          )}
                          description="Select access groups to assign to this team"
                        >
                          {({ value, onChange }) => (
                            <AccessGroupSelector
                              value={value}
                              onChange={onChange}
                              placeholder="Select access groups (optional)"
                            />
                          )}
                        </FormField>
                        <FormField
                          control={form.control}
                          name="allowed_vector_store_ids"
                          className="mt-8"
                          label={labelWithHint(
                            "Allowed Vector Stores",
                            "Select which vector stores this team can access by default. Leave empty for access to all vector stores",
                          )}
                          description="Select vector stores this team can access. Leave empty for access to all vector stores"
                        >
                          {({ value, onChange }) => (
                            <VectorStoreSelector
                              onChange={onChange}
                              value={value}
                              accessToken={accessToken || ""}
                              placeholder="Select vector stores (optional)"
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
                                  "Allowed Pass Through Routes",
                                  "Premium feature - Upgrade to set allowed pass through routes",
                                )
                              : !isProxyAdminRole(userRole || "")
                                ? labelWithHint(
                                    "Allowed Pass Through Routes",
                                    "Only proxy admins can set allowed pass through routes",
                                  )
                                : "Allowed Pass Through Routes"
                          }
                        >
                          {({ value, onChange }) => (
                            <PassThroughRoutesSelector
                              value={value}
                              onChange={onChange}
                              accessToken={accessToken || ""}
                              placeholder="Select pass through routes (optional)"
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
                      <b>MCP Settings</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="allowed_mcp_servers_and_groups"
                        className="mt-4"
                        label={labelWithHint(
                          "Allowed MCP Servers",
                          "Select which MCP servers or access groups this team can access",
                        )}
                        description="Select MCP servers or access groups this team can access"
                      >
                        {({ value, onChange }) => (
                          <MCPServerSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder="Select MCP servers or access groups (optional)"
                            allowAllProxyMcpServers={isProxyAdminRole(userRole || "")}
                          />
                        )}
                      </FormField>

                      <div className="mt-6">
                        <MCPToolPermissions
                          accessToken={accessToken || ""}
                          selectedServers={watchedMcpSelection?.servers || []}
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
                      <b>Agent Settings</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="allowed_agents_and_groups"
                        className="mt-4"
                        label={labelWithHint(
                          "Allowed Agents",
                          "Select which agents or access groups this team can access",
                        )}
                        description="Select agents or access groups this team can access"
                      >
                        {({ value, onChange }) => (
                          <AgentSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder="Select agents or access groups (optional)"
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
                      <b>Search Tool Settings</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <FormField
                        control={form.control}
                        name="object_permission_search_tools"
                        className="mt-4"
                        label={labelWithHint(
                          "Allowed Search Tools",
                          "Select which search tools this team can access. Leave empty to allow all search tools.",
                        )}
                        description="Restrict which configured search tools keys on this team may call."
                      >
                        {({ value, onChange }) => (
                          <SearchToolSelector
                            onChange={onChange}
                            value={value}
                            accessToken={accessToken || ""}
                            placeholder="Select search tools (optional, empty = all allowed)"
                          />
                        )}
                      </FormField>
                    </CollapsibleContent>
                  </Collapsible>

                  <Collapsible className="mt-8 mb-8 overflow-hidden rounded-lg border">
                    <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
                      <b>Logging Settings</b>
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
                      <b>Router Settings</b>
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
                      <b>Model Aliases</b>
                      <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="px-4 pb-3">
                      <div className="mt-4">
                        <p className="mb-4 block text-sm text-muted-foreground">
                          Create custom aliases for models that can be used by team members in API calls. This allows
                          you to create shortcuts for specific models.
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
                    Create Team
                  </UIButton>
                </div>
              </form>
            </TooltipProvider>
          </DialogContent>
        </Dialog>
      )}
    </main>
  );
};

export default Teams;
