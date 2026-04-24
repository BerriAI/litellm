import React, { useEffect, useState, useMemo } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { useQueryClient } from "@tanstack/react-query";
import { Info, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import {
  fetchAvailableModelsForTeamOrKey,
  getModelDisplayName,
  unfurlWildcardModelsInList,
} from "@/components/key_team_helpers/fetch_available_models_team_key";
import NumericalInput from "@/components/shared/numerical_input";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import AgentSelector from "@/components/agent_management/AgentSelector";
import PremiumLoggingSettings from "@/components/common_components/PremiumLoggingSettings";
import ModelAliasManager from "@/components/common_components/ModelAliasManager";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  fetchMCPAccessGroups,
  getGuardrailsList,
  getPoliciesList,
  Organization,
  Team,
  teamCreateCall,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { organizationKeys } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";

interface ModelAliases {
  [key: string]: string;
}

interface CreateTeamModalProps {
  isTeamModalVisible: boolean;
  handleOk: () => void;
  handleCancel: () => void;
  currentOrg: Organization | null;
  organizations: Organization[] | null;
  teams: Team[] | null;
  setTeams: (teams: Team[] | null) => void;
  modelAliases: ModelAliases;
  setModelAliases: (modelAliases: ModelAliases) => void;
  loggingSettings: any[];
  setLoggingSettings: (loggingSettings: any[]) => void;
  setIsTeamModalVisible: (isTeamModalVisible: boolean) => void;
}

type CreateTeamFormValues = {
  team_alias: string;
  organization_id: string | null;
  models: string[];
  default_team_member_models: string[];
  team_member_budget?: number | null;
  team_member_key_duration?: string;
  team_member_rpm_limit?: number | null;
  team_member_tpm_limit?: number | null;
  max_budget?: number | null;
  budget_duration?: string;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  team_id?: string;
  metadata?: string;
  secret_manager_settings?: string;
  guardrails: string[];
  disable_global_guardrails: boolean;
  policies: string[];
  allowed_vector_store_ids: string[];
  allowed_mcp_servers_and_groups?: {
    servers: string[];
    accessGroups: string[];
    toolsets?: string[];
    toolPermissions?: any;
  };
  mcp_tool_permissions?: Record<string, any>;
  allowed_agents_and_groups?: {
    agents: string[];
    accessGroups: string[];
  };
};

const getOrganizationModels = (
  organization: Organization | null,
  userModels: string[],
) => {
  let tempModelsToPick: string[] = [];

  if (organization) {
    if (organization.models.length > 0) {
      tempModelsToPick = organization.models;
    } else {
      tempModelsToPick = userModels;
    }
  } else {
    tempModelsToPick = userModels;
  }

  return unfurlWildcardModelsInList(tempModelsToPick, userModels);
};

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="h-3.5 w-3.5 text-muted-foreground ml-1 inline-flex" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

function TagMultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { value: string; label: string }[];
  placeholder: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );
  const [draft, setDraft] = useState("");

  const addValue = (v: string) => {
    const trimmed = v.trim();
    if (!trimmed) return;
    if (selected.includes(trimmed)) return;
    onChange([...selected, trimmed]);
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-col gap-2 sm:flex-row">
        <Select
          value=""
          onValueChange={(v) => {
            if (v) addValue(v);
          }}
        >
          <SelectTrigger className="flex-1">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {remaining.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No options
              </div>
            ) : (
              remaining.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
        <Input
          className="flex-1"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addValue(draft);
              setDraft("");
            }
          }}
          placeholder="Type and press Enter to add"
        />
      </div>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge key={v} variant="secondary" className="gap-1">
              <span>{v}</span>
              <button
                type="button"
                aria-label={`Remove ${v}`}
                onClick={() => onChange(selected.filter((x) => x !== v))}
                className="ml-1"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function ModelMultiSelect({
  value,
  onChange,
  options,
  placeholder,
  includeAllProxyModels,
  getDisplay = (v) => v,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: string[];
  placeholder: string;
  includeAllProxyModels?: boolean;
  getDisplay?: (v: string) => string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const fullOptions = useMemo(() => {
    const opts = options.map((v) => ({ value: v, label: getDisplay(v) }));
    if (includeAllProxyModels) {
      return [{ value: "all-proxy-models", label: "All Proxy Models" }, ...opts];
    }
    return opts;
  }, [options, includeAllProxyModels, getDisplay]);

  const remaining = useMemo(
    () => fullOptions.filter((o) => !selected.includes(o.value)),
    [fullOptions, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more models
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = fullOptions.find((o) => o.value === v);
            return (
              <Badge key={v} variant="secondary" className="gap-1">
                <span>{opt?.label ?? v}</span>
                <button
                  type="button"
                  aria-label={`Remove ${v}`}
                  onClick={() => onChange(selected.filter((x) => x !== v))}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

const CreateTeamModal = ({
  isTeamModalVisible,
  handleCancel,
  currentOrg,
  organizations,
  teams,
  setTeams,
  modelAliases,
  setModelAliases,
  loggingSettings,
  setLoggingSettings,
  setIsTeamModalVisible,
}: CreateTeamModalProps) => {
  const { userId: userID, userRole, accessToken, premiumUser } = useAuthorized();
  const queryClient = useQueryClient();
  const [userModels, setUserModels] = useState<string[]>([]);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] =
    useState<Organization | null>(null);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);

  const form = useForm<CreateTeamFormValues>({
    defaultValues: {
      team_alias: "",
      organization_id: currentOrg ? currentOrg.organization_id : null,
      models: [],
      default_team_member_models: [],
      team_member_budget: null,
      team_member_key_duration: "",
      team_member_rpm_limit: null,
      team_member_tpm_limit: null,
      max_budget: null,
      budget_duration: undefined,
      tpm_limit: null,
      rpm_limit: null,
      team_id: "",
      metadata: "",
      secret_manager_settings: "",
      guardrails: [],
      disable_global_guardrails: false,
      policies: [],
      allowed_vector_store_ids: [],
      allowed_mcp_servers_and_groups: undefined,
      mcp_tool_permissions: {},
      allowed_agents_and_groups: undefined,
    },
  });
  const {
    register,
    handleSubmit,
    control,
    watch,
    reset,
    setValue,
    formState: { errors },
  } = form;

  const watchedModels = watch("models");

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }
        const models = await fetchAvailableModelsForTeamOrKey(
          userID,
          userRole,
          accessToken,
        );
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole, teams]);

  useEffect(() => {
    const models = getOrganizationModels(currentOrgForCreateTeam, userModels);
    setModelsToPick(models);
    setValue("models", []);
  }, [currentOrgForCreateTeam, userModels, setValue]);

  const fetchMcpAccessGroupsFn = async () => {
    try {
      if (accessToken == null) {
        return;
      }
      await fetchMCPAccessGroups(accessToken);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

  useEffect(() => {
    fetchMcpAccessGroupsFn();
  }, [accessToken]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name,
        );
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
        const policyNames = response.policies.map(
          (p: { policy_name: string }) => p.policy_name,
        );
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    fetchGuardrails();
    fetchPolicies();
  }, [accessToken]);

  const handleCreate = async (rawValues: CreateTeamFormValues) => {
    const formValues: Record<string, any> = { ...rawValues };
    try {
      if (accessToken != null) {
        const newTeamAlias = formValues?.team_alias;
        const existingTeamAliases = teams?.map((t) => t.team_alias) ?? [];
        let organizationId =
          formValues?.organization_id || currentOrg?.organization_id;
        if (organizationId === "" || typeof organizationId !== "string") {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }

        if (existingTeamAliases.includes(newTeamAlias)) {
          throw new Error(
            `Team alias ${newTeamAlias} already exists, please pick another alias`,
          );
        }

        NotificationsManager.info("Creating Team");

        if (loggingSettings.length > 0) {
          let metadata: Record<string, any> = {};
          if (formValues.metadata) {
            try {
              metadata = JSON.parse(formValues.metadata);
            } catch (e) {
              console.warn(
                "Invalid JSON in metadata field, starting with empty object",
              );
            }
          }

          metadata = {
            ...metadata,
            logging: loggingSettings.filter((config) => config.callback_name),
          };

          formValues.metadata = JSON.stringify(metadata);
        }

        if (formValues.secret_manager_settings) {
          if (typeof formValues.secret_manager_settings === "string") {
            if (formValues.secret_manager_settings.trim() === "") {
              delete formValues.secret_manager_settings;
            } else {
              try {
                formValues.secret_manager_settings = JSON.parse(
                  formValues.secret_manager_settings,
                );
              } catch (e) {
                throw new Error(
                  "Failed to parse secret manager settings: " + e,
                );
              }
            }
          }
        }

        if (
          (formValues.allowed_vector_store_ids &&
            formValues.allowed_vector_store_ids.length > 0) ||
          (formValues.allowed_mcp_servers_and_groups &&
            (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.accessGroups?.length >
                0 ||
              formValues.allowed_mcp_servers_and_groups.toolPermissions))
        ) {
          formValues.object_permission = {};
          if (
            formValues.allowed_vector_store_ids &&
            formValues.allowed_vector_store_ids.length > 0
          ) {
            formValues.object_permission.vector_stores =
              formValues.allowed_vector_store_ids;
            delete formValues.allowed_vector_store_ids;
          }
          if (formValues.allowed_mcp_servers_and_groups) {
            const { servers, accessGroups } =
              formValues.allowed_mcp_servers_and_groups;
            if (servers && servers.length > 0) {
              formValues.object_permission.mcp_servers = servers;
            }
            if (accessGroups && accessGroups.length > 0) {
              formValues.object_permission.mcp_access_groups = accessGroups;
            }
            delete formValues.allowed_mcp_servers_and_groups;
          }

          if (
            formValues.mcp_tool_permissions &&
            Object.keys(formValues.mcp_tool_permissions).length > 0
          ) {
            if (!formValues.object_permission) {
              formValues.object_permission = {};
            }
            formValues.object_permission.mcp_tool_permissions =
              formValues.mcp_tool_permissions;
            delete formValues.mcp_tool_permissions;
          }

          if (formValues.allowed_agents_and_groups) {
            const { agents, accessGroups } =
              formValues.allowed_agents_and_groups;
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
        }

        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
        }

        const response: any = await teamCreateCall(accessToken, formValues);
        queryClient.invalidateQueries({ queryKey: organizationKeys.all });
        if (teams !== null) {
          setTeams([...teams, response]);
        } else {
          setTeams([response]);
        }
        NotificationsManager.success("Team created");
        reset();
        setLoggingSettings([]);
        setModelAliases({});
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      NotificationsManager.fromBackend("Error creating the team: " + error);
    }
  };

  const onOpenChange = (open: boolean) => {
    if (!open) {
      handleCancel();
    }
  };

  const defaultMemberModelOpts =
    watchedModels && watchedModels.length > 0 ? watchedModels : modelsToPick;

  return (
    <Dialog open={isTeamModalVisible} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-[1000px] max-h-[90vh] overflow-y-auto"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Create Team</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleSubmit(handleCreate)} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
              <Label htmlFor="team_alias" className="sm:pt-2">
                Team Name <span className="text-destructive">*</span>
              </Label>
              <div>
                <Input
                  id="team_alias"
                  data-testid="team-name-input"
                  {...register("team_alias", {
                    required: "Please input a team name",
                  })}
                />
                {errors.team_alias && (
                  <p className="text-sm text-destructive mt-1">
                    {errors.team_alias.message as string}
                  </p>
                )}
              </div>

              <Label htmlFor="organization_id" className="sm:pt-2">
                <span>
                  Organization
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-muted-foreground ml-1 inline-flex" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        Organizations can have multiple teams. Learn more about{" "}
                        <a
                          href="https://docs.litellm.ai/docs/proxy/user_management_heirarchy"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline"
                          onClick={(e) => e.stopPropagation()}
                        >
                          user management hierarchy
                        </a>
                        .
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </span>
              </Label>
              <Controller
                control={control}
                name="organization_id"
                render={({ field }) => (
                  <Select
                    value={field.value ?? ""}
                    onValueChange={(value) => {
                      const next = value === "__none__" ? null : value;
                      field.onChange(next);
                      setCurrentOrgForCreateTeam(
                        organizations?.find(
                          (org) => org.organization_id === next,
                        ) || null,
                      );
                    }}
                  >
                    <SelectTrigger id="organization_id">
                      <SelectValue placeholder="Search or select an Organization" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__">(none)</SelectItem>
                      {organizations?.map((org) => (
                        <SelectItem
                          key={org.organization_id}
                          value={org.organization_id}
                        >
                          <span className="font-medium">
                            {org.organization_alias}
                          </span>{" "}
                          <span className="text-muted-foreground">
                            ({org.organization_id})
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              />

              <Label className="sm:pt-2">
                <span>
                  Models
                  <InfoTip>
                    These are the models that your selected team has access to
                  </InfoTip>
                </span>
              </Label>
              <Controller
                control={control}
                name="models"
                render={({ field }) => (
                  <div data-testid="team-models-select">
                    <ModelMultiSelect
                      value={field.value ?? []}
                      onChange={field.onChange}
                      options={modelsToPick}
                      placeholder="Select models"
                      includeAllProxyModels
                      getDisplay={getModelDisplayName}
                    />
                  </div>
                )}
              />
            </div>

            <Accordion type="multiple" className="mt-8 mb-8">
              <AccordionItem value="team-member-settings">
                <AccordionTrigger>
                  <b>Team Member Settings</b>
                </AccordionTrigger>
                <AccordionContent>
                  <p className="text-xs text-muted-foreground mb-4">
                    Optional defaults applied when members join this team. All
                    fields can be overridden per member.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
                    <Label className="sm:pt-2">
                      <span>
                        Default Model Access
                        <InfoTip>
                          Optional. If set, new members can only access these
                          models by default. Must be a subset of the team&apos;s
                          models. Leave empty to give all members access to all
                          team models.
                        </InfoTip>
                      </span>
                    </Label>
                    <Controller
                      control={control}
                      name="default_team_member_models"
                      render={({ field }) => (
                        <ModelMultiSelect
                          value={field.value ?? []}
                          onChange={field.onChange}
                          options={defaultMemberModelOpts}
                          placeholder="Leave empty — all team models accessible to every member"
                          getDisplay={getModelDisplayName}
                        />
                      )}
                    />

                    <Label htmlFor="team_member_budget" className="sm:pt-2">
                      Default Member Budget (USD)
                    </Label>
                    <Controller
                      control={control}
                      name="team_member_budget"
                      render={({ field }) => (
                        <NumericalInput
                          step={0.01}
                          precision={2}
                          style={{ width: 200 }}
                          value={field.value ?? undefined}
                          onChange={(v: number | null) =>
                            field.onChange(v !== null ? Number(v) : undefined)
                          }
                        />
                      )}
                    />

                    <Label
                      htmlFor="team_member_key_duration"
                      className="sm:pt-2"
                    >
                      Default Key Duration (eg: 1d, 1mo)
                    </Label>
                    <Input
                      id="team_member_key_duration"
                      placeholder="e.g., 30d"
                      {...register("team_member_key_duration")}
                    />

                    <Label htmlFor="team_member_rpm_limit" className="sm:pt-2">
                      Default RPM Limit
                    </Label>
                    <Controller
                      control={control}
                      name="team_member_rpm_limit"
                      render={({ field }) => (
                        <NumericalInput
                          step={1}
                          style={{ width: 400 }}
                          value={field.value ?? undefined}
                          onChange={field.onChange}
                        />
                      )}
                    />

                    <Label htmlFor="team_member_tpm_limit" className="sm:pt-2">
                      Default TPM Limit
                    </Label>
                    <Controller
                      control={control}
                      name="team_member_tpm_limit"
                      render={({ field }) => (
                        <NumericalInput
                          step={1}
                          style={{ width: 400 }}
                          value={field.value ?? undefined}
                          onChange={field.onChange}
                        />
                      )}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
              <Label htmlFor="max_budget" className="sm:pt-2">
                Max Budget (USD)
              </Label>
              <Controller
                control={control}
                name="max_budget"
                render={({ field }) => (
                  <NumericalInput
                    step={0.01}
                    precision={2}
                    style={{ width: 200 }}
                    value={field.value ?? undefined}
                    onChange={field.onChange}
                  />
                )}
              />

              <Label htmlFor="budget_duration" className="sm:pt-2 mt-8 sm:mt-0">
                Reset Budget
              </Label>
              <Controller
                control={control}
                name="budget_duration"
                render={({ field }) => (
                  <Select
                    value={field.value ?? ""}
                    onValueChange={field.onChange}
                  >
                    <SelectTrigger id="budget_duration">
                      <SelectValue placeholder="n/a" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="24h">daily</SelectItem>
                      <SelectItem value="7d">weekly</SelectItem>
                      <SelectItem value="30d">monthly</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />

              <Label htmlFor="tpm_limit" className="sm:pt-2">
                Tokens per minute Limit (TPM)
              </Label>
              <Controller
                control={control}
                name="tpm_limit"
                render={({ field }) => (
                  <NumericalInput
                    step={1}
                    style={{ width: 400 }}
                    value={field.value ?? undefined}
                    onChange={field.onChange}
                  />
                )}
              />

              <Label htmlFor="rpm_limit" className="sm:pt-2">
                Requests per minute Limit (RPM)
              </Label>
              <Controller
                control={control}
                name="rpm_limit"
                render={({ field }) => (
                  <NumericalInput
                    step={1}
                    style={{ width: 400 }}
                    value={field.value ?? undefined}
                    onChange={field.onChange}
                  />
                )}
              />
            </div>

            <Accordion
              type="multiple"
              className="mt-8 mb-8"
              onClick={() => {
                if (!mcpAccessGroupsLoaded) {
                  fetchMcpAccessGroupsFn();
                  setMcpAccessGroupsLoaded(true);
                }
              }}
            >
              <AccordionItem value="additional-settings">
                <AccordionTrigger>
                  <b>Additional Settings</b>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
                    <Label htmlFor="team_id" className="sm:pt-2">
                      Team ID
                    </Label>
                    <div>
                      <Input
                        id="team_id"
                        {...register("team_id", {
                          onChange: (e) => {
                            e.target.value = e.target.value.trim();
                          },
                        })}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        ID of the team you want to create. If not provided, it
                        will be generated automatically.
                      </p>
                    </div>

                    <Label htmlFor="metadata" className="sm:pt-2">
                      Metadata
                    </Label>
                    <div>
                      <Textarea
                        id="metadata"
                        rows={4}
                        {...register("metadata")}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Additional team metadata. Enter metadata as JSON object.
                      </p>
                    </div>

                    <Label htmlFor="secret_manager_settings" className="sm:pt-2">
                      Secret Manager Settings
                    </Label>
                    <div>
                      <Textarea
                        id="secret_manager_settings"
                        rows={4}
                        placeholder='{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
                        disabled={!premiumUser}
                        {...register("secret_manager_settings", {
                          validate: (value) => {
                            if (!value) return true;
                            try {
                              JSON.parse(value);
                              return true;
                            } catch {
                              return "Please enter valid JSON";
                            }
                          },
                        })}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        {premiumUser
                          ? "Enter secret manager configuration as a JSON object."
                          : "Premium feature - Upgrade to manage secret manager settings."}
                      </p>
                      {errors.secret_manager_settings && (
                        <p className="text-sm text-destructive mt-1">
                          {errors.secret_manager_settings.message as string}
                        </p>
                      )}
                    </div>

                    <Label className="sm:pt-2">
                      <span>
                        Guardrails
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <a
                                href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <Info className="h-3.5 w-3.5 text-muted-foreground ml-1 inline-flex" />
                              </a>
                            </TooltipTrigger>
                            <TooltipContent>
                              Setup your first guardrail
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="guardrails"
                        render={({ field }) => (
                          <TagMultiSelect
                            value={field.value ?? []}
                            onChange={field.onChange}
                            options={guardrailsList.map((name) => ({
                              value: name,
                              label: name,
                            }))}
                            placeholder="Select existing guardrails"
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Select existing guardrails or enter new ones
                      </p>
                    </div>

                    <Label
                      htmlFor="disable_global_guardrails"
                      className="sm:pt-2"
                    >
                      <span>
                        Disable Global Guardrails
                        <InfoTip>
                          When enabled, this team will bypass any guardrails
                          configured to run on every request (global guardrails)
                        </InfoTip>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="disable_global_guardrails"
                        render={({ field }) => (
                          <Switch
                            id="disable_global_guardrails"
                            checked={!!field.value}
                            onCheckedChange={field.onChange}
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Bypass global guardrails for this team
                      </p>
                    </div>

                    <Label className="sm:pt-2">
                      <span>
                        Policies
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <a
                                href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <Info className="h-3.5 w-3.5 text-muted-foreground ml-1 inline-flex" />
                              </a>
                            </TooltipTrigger>
                            <TooltipContent>
                              Apply policies to this team to control guardrails
                              and other settings
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="policies"
                        render={({ field }) => (
                          <TagMultiSelect
                            value={field.value ?? []}
                            onChange={field.onChange}
                            options={policiesList.map((name) => ({
                              value: name,
                              label: name,
                            }))}
                            placeholder="Select existing policies"
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Select existing policies or enter new ones
                      </p>
                    </div>

                    <Label className="sm:pt-2">
                      <span>
                        Allowed Vector Stores
                        <InfoTip>
                          Select which vector stores this team can access by
                          default. Leave empty for access to all vector stores
                        </InfoTip>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="allowed_vector_store_ids"
                        render={({ field }) => (
                          <VectorStoreSelector
                            onChange={(values: string[]) =>
                              field.onChange(values)
                            }
                            value={field.value}
                            accessToken={accessToken || ""}
                            placeholder="Select vector stores (optional)"
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Select vector stores this team can access. Leave empty
                        for access to all vector stores
                      </p>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <Accordion type="multiple" className="mt-8 mb-8">
              <AccordionItem value="mcp-settings">
                <AccordionTrigger>
                  <b>MCP Settings</b>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
                    <Label className="sm:pt-2">
                      <span>
                        Allowed MCP Servers
                        <InfoTip>
                          Select which MCP servers or access groups this team
                          can access
                        </InfoTip>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="allowed_mcp_servers_and_groups"
                        render={({ field }) => (
                          <MCPServerSelector
                            onChange={(val: any) => field.onChange(val)}
                            value={field.value}
                            accessToken={accessToken || ""}
                            placeholder="Select MCP servers or access groups (optional)"
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Select MCP servers or access groups this team can access
                      </p>
                    </div>
                  </div>

                  <div className="mt-6">
                    <Controller
                      control={control}
                      name="mcp_tool_permissions"
                      render={({ field }) => (
                        <MCPToolPermissions
                          accessToken={accessToken || ""}
                          selectedServers={
                            watch("allowed_mcp_servers_and_groups")?.servers ||
                            []
                          }
                          toolPermissions={field.value || {}}
                          onChange={(toolPerms) => field.onChange(toolPerms)}
                        />
                      )}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <Accordion type="multiple" className="mt-8 mb-8">
              <AccordionItem value="agent-settings">
                <AccordionTrigger>
                  <b>Agent Settings</b>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-x-4 gap-y-4 items-start">
                    <Label className="sm:pt-2">
                      <span>
                        Allowed Agents
                        <InfoTip>
                          Select which agents or access groups this team can
                          access
                        </InfoTip>
                      </span>
                    </Label>
                    <div>
                      <Controller
                        control={control}
                        name="allowed_agents_and_groups"
                        render={({ field }) => (
                          <AgentSelector
                            onChange={(val: any) => field.onChange(val)}
                            value={field.value}
                            accessToken={accessToken || ""}
                            placeholder="Select agents or access groups (optional)"
                          />
                        )}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Select agents or access groups this team can access
                      </p>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <Accordion type="multiple" className="mt-8 mb-8">
              <AccordionItem value="logging-settings">
                <AccordionTrigger>
                  <b>Logging Settings</b>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="mt-4">
                    <PremiumLoggingSettings
                      value={loggingSettings}
                      onChange={setLoggingSettings}
                      premiumUser={premiumUser}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <Accordion type="multiple" className="mt-8 mb-8">
              <AccordionItem value="model-aliases">
                <AccordionTrigger>
                  <b>Model Aliases</b>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="mt-4">
                    <p className="text-sm text-muted-foreground mb-4">
                      Create custom aliases for models that can be used by team
                      members in API calls. This allows you to create shortcuts
                      for specific models.
                    </p>
                    <ModelAliasManager
                      accessToken={accessToken || ""}
                      initialModelAliases={modelAliases}
                      onAliasUpdate={setModelAliases}
                      showExampleConfig={false}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            <div className="flex justify-end mt-4">
              <Button type="submit" data-testid="create-team-submit">
                Create Team
              </Button>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default CreateTeamModal;
