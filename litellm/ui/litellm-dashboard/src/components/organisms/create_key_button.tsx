"use client";
import { useTerminology } from "@/contexts/TerminologyContext";
import { keyKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { parseTeamAdminEditableFields } from "@/components/team/teamAdminEditAccess";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { useQueryClient } from "@tanstack/react-query";
import type { ParseKeys } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";
import { Badge } from "@/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { ChevronDown, Info } from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { type Control, useForm, useWatch, type UseFormSetValue } from "react-hook-form";
import { isProxyAdminRole, rolesWithWriteAccess } from "../../utils/roles";
import AgentSelector from "../agent_management/AgentSelector";
import SkillSelector from "../skills/SkillSelector";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import SchemaFormFields from "../common_components/check_openapi_schema";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import ModelAliasManager from "../common_components/ModelAliasManager";
import { StorePromptsMetadataToggle } from "../common_components/StorePromptsToggle";
import {
  MountedFormField,
  MountedFormProvider,
  projectMountedValues,
  useMountedName,
  useMountRegistry,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import PremiumLoggingSettings from "../common_components/PremiumLoggingSettings";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import RouterSettingsAccordion, {
  RouterSettingsAccordionRef,
  RouterSettingsAccordionValue,
} from "../common_components/RouterSettingsAccordion";
import TeamDropdown from "../common_components/team_dropdown";
import OrganizationDropdown from "../common_components/OrganizationDropdown";
import ProjectDropdown from "../common_components/ProjectDropdown";
import { CreateUserButton } from "../CreateUserButton";
import { moveMetadataBudgetAlertsToRows } from "../templates/KeyEditViewControls";
import { BudgetAlertEmailsEditor, type BudgetAlertEmailsEditorText } from "../key_team_helpers/BudgetAlertEmailsEditor";
import {
  type BudgetAlertRow,
  budgetAlertRowsSchema,
  KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
} from "../key_team_helpers/budgetAlertEmails";
import { BudgetFallbacksEditor } from "../key_team_helpers/BudgetFallbacksEditor";
import { END_USER_BUDGET_HINT, EndUserBudgetSelect } from "../key_team_helpers/EndUserBudgetSelect";
import { BudgetWindowEntry, BudgetWindowsEditor } from "../key_team_helpers/BudgetWindowsEditor";
import { ModelMaxBudget, ModelMaxBudgetEditor } from "../key_team_helpers/ModelMaxBudgetEditor";
import { TagRateLimitEditor, TagRateLimitEntry } from "../key_team_helpers/TagRateLimitEditor";
import {
  excludeProxyWideSentinel,
  getModelDisplayName,
  hasAllModelsSentinel,
} from "../key_team_helpers/fetch_available_models_team_key";
import { Team } from "../key_team_helpers/key_list";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import { toast } from "@/lib/toast";
import {
  getAgentsList,
  getGuardrailsList,
  getPoliciesList,
  getPossibleUserRoles,
  getPromptsList,
  keyCreateCall,
  keyCreateServiceAccountCall,
  modelAvailableCall,
  proxyBaseUrl,
  userFilterUICall,
} from "../networking";
import CreatedKeyDisplay from "../shared/CreatedKeyDisplay";
import NumericalInput from "../shared/numerical_input";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import {
  administersTeam,
  anotherUserScope,
  effectiveKeyOwner,
  isTeamMemberOwner,
  teamMemberOwnerOptions,
  teamMemberOwnerPlaceholder,
} from "./createKeyOwnerAccess";
import { buildKeyCreatePayload, type KeyCreateInput } from "./createKeyPayload";
import { simplifyKeyGenerateError } from "./utils";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const GENERATE_KEY_EXCLUDED_FIELDS = [
  "key_alias",
  "team_id",
  "organization_id",
  "models",
  "duration",
  "metadata",
  "tags",
  "guardrails",
  "max_budget",
  "budget_duration",
  "tpm_limit",
  "rpm_limit",
  "tpd_limit",
];
const GENERATE_KEY_EXCLUDED_FIELDS_WITH_KEY = [...GENERATE_KEY_EXCLUDED_FIELDS, "key"];

const KEY_TYPE_OPTIONS: { value: string; label: ParseKeys; hint: ParseKeys }[] = [
  { value: "llm_api", label: "createKey.keyTypes.llmApi.label", hint: "createKey.keyTypes.llmApi.hint" },
  { value: "management", label: "createKey.keyTypes.management.label", hint: "createKey.keyTypes.management.hint" },
  { value: "default", label: "createKey.keyTypes.fullAccess.label", hint: "createKey.keyTypes.fullAccess.hint" },
];

const KEY_OWNER_LABEL_CLASS = "flex items-center gap-2 text-sm font-normal text-foreground";

const SECTION_HEADER_CLASS = "group/section flex w-full items-center justify-between px-4 py-3 text-left";
const SECTION_CHEVRON_CLASS =
  "size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180";

type FieldWrite = (value: unknown) => void;

type McpSelectorValue = { servers: string[]; accessGroups: string[]; toolsets?: string[] };

type AgentSelectorValue = { agents: string[]; accessGroups: string[] };

const MountedBudgetAlertEmailsEditor = (props: {
  readonly control: Control<MountedFormValues>;
  readonly label: React.ReactNode;
  readonly text: BudgetAlertEmailsEditorText;
}) => {
  useMountedName(KEY_MAX_BUDGET_ALERT_EMAILS_KEY);
  return <BudgetAlertEmailsEditor {...props} name={KEY_MAX_BUDGET_ALERT_EMAILS_KEY} />;
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || value === "";

const requiredRule = (required: boolean, message: string) => ({
  validate: (value: unknown) => (required && isBlank(value) ? message : true),
});

const ceilingRule = (ceiling: number | null | undefined, message: (limit: number) => string) => ({
  validate: (value: unknown) =>
    value && ceiling !== null && ceiling !== undefined && (value as number) > ceiling ? message(ceiling) : true,
});

interface McpToolPermissionsFieldProps {
  readonly accessToken: string;
  readonly control: Control<MountedFormValues>;
  readonly setValue: UseFormSetValue<MountedFormValues>;
}

const McpToolPermissionsField: React.FC<McpToolPermissionsFieldProps> = ({ accessToken, control, setValue }) => {
  const selection = useWatch({ control, name: "allowed_mcp_servers_and_groups" }) as
    | { servers?: string[]; accessGroups?: string[]; toolsets?: string[] }
    | undefined;
  const toolPermissions = useWatch({ control, name: "mcp_tool_permissions" }) as Record<string, string[]> | undefined;

  return (
    <div className="mt-6">
      <MCPToolPermissions
        accessToken={accessToken}
        selectedServers={selection?.servers || []}
        selectedAccessGroups={selection?.accessGroups || []}
        selectedToolsets={selection?.toolsets || []}
        toolPermissions={toolPermissions || {}}
        onChange={(toolPerms) => setValue("mcp_tool_permissions", toolPerms)}
      />
    </div>
  );
};

/**
 * Interface for pre-filling the create key form from URL parameters
 */
export interface CreateKeyPrefillData {
  owned_by?: "you" | "service_account" | "another_user";
  team_id?: string;
  key_alias?: string;
  models?: string[];
  key_type?: "default" | "llm_api" | "management";
}

interface CreateKeyProps {
  team: Team | null;
  data: any[] | null;
  teams: Team[] | null;
  addKey: (data: any) => void;
  autoOpenCreate?: boolean;
  prefillData?: CreateKeyPrefillData;
}

interface User {
  user_id: string;
  user_email: string | null;
  role?: string;
}

export const fetchTeamModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  teamID: string | null,
): Promise<string[]> => {
  try {
    if (userID === null || userRole === null) {
      return [];
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole, true, teamID, true);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      return available_model_names;
    }
    return [];
  } catch (error) {
    console.error("Error fetching user models:", error);
    return [];
  }
};

export const fetchUserModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  setUserModels: (models: string[]) => void,
) => {
  try {
    if (userID === null || userRole === null) {
      return;
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      setUserModels(available_model_names);
    }
  } catch (error) {
    console.error("Error fetching user models:", error);
  }
};

/**
 * ─────────────────────────────────────────────────────────────────────────
 * @deprecated
 * This component is being DEPRECATED in favor of src/app/(dashboard)/virtual-keys/components/CreateKey.tsx
 * Please contribute to the new refactor.
 * ─────────────────────────────────────────────────────────────────────────
 */
const CreateKey: React.FC<CreateKeyProps> = ({ team, teams, data, addKey, autoOpenCreate, prefillData }) => {
  const { accessToken, userId: userID, userRole, premiumUser } = useAuthorized();
  const term = useTerminology();
  const { t } = useTranslation();
  const keyTypeOptions = KEY_TYPE_OPTIONS.map((option) => ({
    ...option,
    label: term(t(option.label)),
    hint: term(t(option.hint)),
  }));
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const canViewPolicies = useCan("viewPolicies");
  const canViewPrompts = useCan("viewPrompts");
  const { data: organizations, isLoading: isOrganizationsLoading } = useOrganizations();
  const { data: projects, isLoading: isProjectsLoading } = useProjects();
  const { data: uiSettingsData } = useUISettings();
  const { data: tagsData } = useTags();
  const enableProjectsUI = Boolean(uiSettingsData?.values?.enable_projects_ui);
  const disableCustomApiKeys = Boolean(uiSettingsData?.values?.disable_custom_api_keys);
  const tagOptions = tagsData ? Object.values(tagsData).map((tag) => ({ value: tag.name, label: tag.name })) : [];
  const queryClient = useQueryClient();
  const [formDefaults] = useState<MountedFormValues>(() => ({
    team_id: team ? team.team_id : null,
    key_type: "llm_api",
    tpm_limit_type: null,
    rpm_limit_type: null,
    mcp_tool_permissions: {},
    duration: "",
  }));
  const form = useForm<MountedFormValues>({
    mode: "onChange",
    shouldUnregister: false,
    defaultValues: formDefaults,
  });
  const registry = useMountRegistry();
  const mountedForm = useMemo(() => ({ control: form.control, registry }), [form.control, registry]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [requestedKeyOwner, setKeyOwner] = useState("you");
  const [hasPrefilled, setHasPrefilled] = useState(false);
  const [pendingPrefillModels, setPendingPrefillModels] = useState<string[] | null>(null);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [selectedCreateKeyTeam, setSelectedCreateKeyTeam] = useState<Team | null>(team);
  const [selectedOrganizationId, setSelectedOrganizationId] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [isCreateUserModalVisible, setIsCreateUserModalVisible] = useState(false);
  const [possibleUIRoles, setPossibleUIRoles] = useState<Record<string, Record<string, string>>>({});
  const [userOptions, setUserOptions] = useState<SearchSelectOption[]>([]);
  const [userSearchLoading, setUserSearchLoading] = useState<boolean>(false);
  const latestUserSearchRef = useRef(0);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>([]);
  const [keyType, setKeyType] = useState<string>("llm_api");
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(false);
  const [rotationInterval, setRotationInterval] = useState<string>("30d");
  const [routerSettings, setRouterSettings] = useState<RouterSettingsAccordionValue | null>(null);
  const routerSettingsRef = useRef<RouterSettingsAccordionRef>(null);
  const [budgetLimits, setBudgetLimits] = useState<BudgetWindowEntry[]>([]);
  const [modelMaxBudget, setModelMaxBudget] = useState<ModelMaxBudget>({});
  const [tagRateLimits, setTagRateLimits] = useState<TagRateLimitEntry[]>([]);
  const [budgetFallbacks, setBudgetFallbacks] = useState<Record<string, string[]>>({});
  const [budgetFallbacksKey, setBudgetFallbacksKey] = useState<number>(0);
  const [routerSettingsKey, setRouterSettingsKey] = useState<number>(0);
  const [agentsList, setAgentsList] = useState<{ agent_id: string; agent_name: string }[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const selectedModels: string[] = (useWatch({ control: form.control, name: "models" }) as string[] | undefined) ?? [];
  const teamAdminEditableFields = parseTeamAdminEditableFields(uiSettingsData?.values);
  const keyOwnerViewer = { userRole, userId: userID, teamAdminEditableFields, teams };
  const ownerScope = anotherUserScope(keyOwnerViewer);
  const keyOwner = effectiveKeyOwner(requestedKeyOwner, ownerScope);
  const ownerLimitedToTeamMembers = keyOwner === "another_user" && ownerScope === "team_members";
  const memberOwnerOptions = teamMemberOwnerOptions(selectedCreateKeyTeam, userID);
  const createKeyOwner = form.watch("user_id");
  useEffect(() => {
    form.setValue("allowed_mcp_servers_and_groups", { servers: [], accessGroups: [] });
    form.setValue("mcp_tool_permissions", {});
  }, [createKeyOwner, form]);
  const ownerRequiresTeam = keyOwner === "service_account" || ownerLimitedToTeamMembers;
  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    setSelectedCreateKeyTeam(null);
    form.reset(formDefaults);
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("llm_api");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
    setSelectedAgentId(null);
    setSelectedOrganizationId(null);
    setSelectedProjectId(null);
    setBudgetLimits([]);
    setTagRateLimits([]);
    setBudgetFallbacks({});
    setBudgetFallbacksKey((k) => k + 1);
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    if (accessToken) {
      getAgentsList(accessToken)
        .then((res) => setAgentsList(res?.agents || []))
        .catch(() => setAgentsList([]));
    }
  }, [accessToken]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    const fetchPrompts = async () => {
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(Array.from(new Set(response.prompts.map((prompt) => prompt.prompt_id))));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    fetchGuardrails();
    if (canViewPolicies) fetchPolicies();
    if (canViewPrompts) fetchPrompts();
  }, [accessToken, canViewPolicies, canViewPrompts]);

  // Fetch possible user roles when component mounts
  useEffect(() => {
    const fetchPossibleRoles = async () => {
      try {
        if (accessToken) {
          // Check if roles are cached in session storage
          const cachedRoles = sessionStorage.getItem("possibleUserRoles");
          if (cachedRoles) {
            setPossibleUIRoles(JSON.parse(cachedRoles));
          } else {
            const availableUserRoles = await getPossibleUserRoles(accessToken);
            sessionStorage.setItem("possibleUserRoles", JSON.stringify(availableUserRoles));
            setPossibleUIRoles(availableUserRoles);
          }
        }
      } catch (error) {
        console.error("Error fetching possible user roles:", error);
      }
    };

    fetchPossibleRoles();
  }, [accessToken]);

  // Auto-open modal and prefill form from URL params (deep link).
  // Guarded by write access so we don't open for read-only users.
  useEffect(() => {
    if (autoOpenCreate && !hasPrefilled && teams && userRole && rolesWithWriteAccess.includes(userRole)) {
      // Open the modal
      setIsModalVisible(true);
      setHasPrefilled(true);

      // Apply prefill data if provided
      if (prefillData) {
        if (prefillData.owned_by) {
          setKeyOwner(prefillData.owned_by);
        }

        // Set team - find the team by ID and set it (only if team exists in user's teams)
        if (prefillData.team_id) {
          const selectedTeam = teams?.find((t) => t.team_id === prefillData.team_id) || null;
          if (selectedTeam) {
            setSelectedCreateKeyTeam(selectedTeam);
            form.setValue("team_id", prefillData.team_id);
          }
          // Silently ignore invalid team_id - don't prefill with a team user doesn't have access to
        }

        // Set key alias
        if (prefillData.key_alias) {
          form.setValue("key_alias", prefillData.key_alias);
        }

        // Defer model selection until we load the allowed model list.
        if (prefillData.models && prefillData.models.length > 0) {
          setPendingPrefillModels(prefillData.models);
        }

        // Set key type
        if (prefillData.key_type) {
          setKeyType(prefillData.key_type);
          form.setValue("key_type", prefillData.key_type);
        }
      }
    }
  }, [autoOpenCreate, prefillData, teams, hasPrefilled, form, userRole]);

  // Check if team selection is required
  const isTeamSelectionRequired = modelsToPick.includes("no-default-models");
  const isFormDisabled = isTeamSelectionRequired && !selectedCreateKeyTeam;

  const handleCreate = async (formValues: MountedFormValues) => {
    try {
      const input: KeyCreateInput = {
        formValues,
        existingKeys: data,
        keyOwner,
        userID,
        selectedAgentId,
        loggingSettings,
        disabledCallbacks,
        autoRotationEnabled,
        rotationInterval,
        modelAliases,
        routerSettings: routerSettingsRef.current?.getValue() ?? routerSettings,
        budgetLimits,
        modelMaxBudget,
        tagRateLimits,
        budgetFallbacks,
      };
      const built = buildKeyCreatePayload(input);
      if (built.kind === "duplicate_alias") {
        throw new Error(
          `Key alias ${built.alias} already exists for team with ID ${built.teamId}, please provide another key alias`,
        );
      }

      toast.info(t("createKey.toasts.makingApiCall"));
      setIsModalVisible(true);

      if (built.kind === "agent_not_selected") {
        toast.fromError(t("createKey.toasts.selectAgent"));
        return;
      }
      const { payload, endpoint } = built;

      const response =
        endpoint === "service_account"
          ? await keyCreateServiceAccountCall(accessToken, payload)
          : await keyCreateCall(accessToken, userID ?? "", payload);

      // Add the data to the state in the parent component
      // Also directly update the keys list in VirtualKeysTable without an API call
      addKey(response);

      // Invalidate and refetch all keys list queries to update the table
      // This will trigger a refetch of all key list queries regardless of pagination
      queryClient.invalidateQueries({ queryKey: keyKeys.lists() });

      setApiKey(response["key"]);
      toast.success(t("createKey.toasts.created"));
      form.reset(formDefaults);
      setBudgetLimits([]);
      setTagRateLimits([]);
      setBudgetFallbacks({});
      setBudgetFallbacksKey((k) => k + 1);
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      const simplifiedError = simplifyKeyGenerateError(error, t);
      toast.fromError(simplifiedError);
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) =>
    void form.handleSubmit(() => {
      const values = projectMountedValues(registry, form.getValues);
      const alertRows = budgetAlertRowsSchema((issue) =>
        issue === "thresholdRange"
          ? t("keyDetail.maxBudgetAlertEmails.thresholdRange")
          : t("keyDetail.maxBudgetAlertEmails.duplicateThreshold"),
      ).safeParse(values[KEY_MAX_BUDGET_ALERT_EMAILS_KEY] ?? []);
      if (!alertRows.success) {
        alertRows.error.issues.forEach((issue) =>
          form.setError([KEY_MAX_BUDGET_ALERT_EMAILS_KEY, ...issue.path].join("."), { message: issue.message }),
        );
        return;
      }
      return handleCreate(values);
    })(event);

  // Fetch available models when team or auth changes.
  // Note: Model prefill from URL params is handled by the useEffect below, which
  // watches for pendingPrefillModels + modelsToPick to both be populated.
  useEffect(() => {
    if (selectedProjectId) {
      // When a project is selected, use the project's models
      const project = projects?.find((p) => p.project_id === selectedProjectId);
      const projectModels = project?.models ?? [];
      setModelsToPick(projectModels);
      form.setValue("models", []);
      return;
    }
    if (userID && userRole && accessToken) {
      fetchTeamModels(userID, userRole, accessToken, selectedCreateKeyTeam?.team_id ?? null).then((models) => {
        const allModels = excludeProxyWideSentinel(
          Array.from(new Set([...(selectedCreateKeyTeam?.models ?? []), ...models])),
        );
        setModelsToPick(allModels);
      });
    }
    // Only clear models if we don't have pending prefill models
    if (!pendingPrefillModels) {
      form.setValue("models", []);
    }
    // Clear MCP server selection when team changes (available servers may differ)
    form.setValue("allowed_mcp_servers_and_groups", { servers: [], accessGroups: [] });
    form.setValue("mcp_tool_permissions", {});
  }, [selectedCreateKeyTeam, selectedProjectId, accessToken, userID, userRole, form]);

  // Apply deferred model prefill once the available model list arrives.
  // This handles timing where prefill data arrives before or after models are fetched.
  useEffect(() => {
    if (!pendingPrefillModels || pendingPrefillModels.length === 0) {
      return;
    }
    if (!modelsToPick || modelsToPick.length === 0) {
      return;
    }

    const validModels = pendingPrefillModels.filter((model) => modelsToPick.includes(model));
    if (validModels.length > 0) {
      form.setValue("models", validModels);
    }
    setPendingPrefillModels(null);
  }, [pendingPrefillModels, modelsToPick, form]);

  // Sync team when project is selected but teams loaded later (race condition)
  useEffect(() => {
    if (!selectedProjectId || !teams) return;
    const project = projects?.find((p) => p.project_id === selectedProjectId);
    if (!project?.team_id) return;
    // If team is already set correctly, skip
    if (selectedCreateKeyTeam?.team_id === project.team_id) return;
    const projectTeam = teams.find((t) => t.team_id === project.team_id) || null;
    if (projectTeam) {
      setSelectedCreateKeyTeam(projectTeam);
      form.setValue("team_id", projectTeam.team_id);
    }
  }, [teams, selectedProjectId, projects]);

  // Add a callback function to handle user creation
  const handleUserCreated = (userId: string) => {
    form.setValue("user_id", userId);
    setIsCreateUserModalVisible(false);
  };

  const fetchUsers = async (searchText: string): Promise<void> => {
    const searchId = latestUserSearchRef.current + 1;
    latestUserSearchRef.current = searchId;
    const isLatestSearch = (): boolean => searchId === latestUserSearchRef.current;

    if (!searchText) {
      setUserOptions([]);
      setUserSearchLoading(false);
      return;
    }

    setUserSearchLoading(true);
    try {
      const params = new URLSearchParams();
      params.append("search", searchText);
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);
      if (!isLatestSearch()) return;

      const data: User[] = response;
      const options: SearchSelectOption[] = data.map((user) => ({
        label: user.user_email ? `${user.user_email} (${user.user_id})` : user.user_id,
        value: user.user_id,
      }));

      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
      if (isLatestSearch()) toast.fromError(t("createKey.toasts.userSearchFailed"));
    } finally {
      if (isLatestSearch()) setUserSearchLoading(false);
    }
  };

  const changeOrganization = (write: FieldWrite) => (orgId: string | null) => {
    write(orgId);
    setSelectedOrganizationId(orgId);
    // Clear team and project when org changes
    setSelectedCreateKeyTeam(null);
    setSelectedProjectId(null);
    form.setValue("team_id", null);
    form.setValue("project_id", null);
  };

  const selectTeam = (team: Team | null) => {
    setSelectedCreateKeyTeam(team);
    setSelectedProjectId(null);
    form.setValue("project_id", null);
    // Auto-populate org from team for non-admin users
    if (team?.organization_id) {
      setSelectedOrganizationId(team.organization_id);
      form.setValue("organization_id", team.organization_id);
    } else if (!team) {
      setSelectedOrganizationId(null);
      form.setValue("organization_id", null);
    }
  };

  const changeKeyOwner = (owner: string) => {
    setKeyOwner(owner);
    const limitsOwnerToTeamMembers = owner === "another_user" && ownerScope === "team_members";
    if (limitsOwnerToTeamMembers && selectedCreateKeyTeam && !administersTeam(selectedCreateKeyTeam, userID)) {
      selectTeam(null);
      form.setValue("team_id", null);
    }
  };

  useEffect(() => {
    if (!ownerLimitedToTeamMembers) return;
    const owner = form.getValues("user_id");
    if (owner != null && !isTeamMemberOwner(owner, selectedCreateKeyTeam, userID)) {
      form.setValue("user_id", null);
    }
  }, [ownerLimitedToTeamMembers, selectedCreateKeyTeam, userID, form]);

  const changeProject = (write: FieldWrite) => (projectId: string | null) => {
    write(projectId);
    if (!projectId) {
      setSelectedProjectId(null);
      setSelectedCreateKeyTeam(null);
      form.setValue("team_id", null);
      return;
    }
    setSelectedProjectId(projectId);
  };

  const modelOptions: MultiSelectOption[] = [
    ...(selectedProjectId === null && selectedCreateKeyTeam
      ? [{ value: "all-team-models", label: t("createKey.models.allTeamModels") }]
      : []),
    ...(selectedProjectId === null && !selectedCreateKeyTeam
      ? [{ value: "all-proxy-models", label: t("createKey.models.allProxyModels") }]
      : []),
    ...modelsToPick.map((model) => ({
      value: model,
      label: getModelDisplayName(model),
      disabled: hasAllModelsSentinel(selectedModels),
    })),
  ];

  const changeKeyType = (write: FieldWrite) => (value: string) => {
    write(value);
    setKeyType(value);
    // Clear models field and disable if management or read_only
    if (value === "management" || value === "read_only") {
      form.setValue("models", []);
    }
  };

  const memberOwnerPlaceholder = teamMemberOwnerPlaceholder(memberOwnerOptions, selectedCreateKeyTeam, t);
  const requiredTeamHelp = ownerLimitedToTeamMembers
    ? t("createKey.team.helpRequiredMember")
    : t("createKey.team.helpRequired");
  const teamFieldHelp = ownerRequiresTeam ? requiredTeamHelp : "";
  const ownerField = keyOwner === "another_user" && (
    <MountedFormField
      label={
        <span>
          {t("createKey.userId.label")}{" "}
          <SimpleTooltip content={t("createKey.userId.tooltip")}>
            <Info className="ml-1 inline size-3.5 align-text-bottom" />
          </SimpleTooltip>
        </span>
      }
      name="user_id"
      className="mt-4"
      required
      rules={requiredRule(keyOwner === "another_user", t("createKey.userId.required"))}
    >
      {(control) =>
        ownerLimitedToTeamMembers ? (
          <SearchSelect
            inputId={control.id}
            options={memberOwnerOptions}
            value={typeof control.value === "string" ? control.value : null}
            onValueChange={control.onChange}
            disabled={memberOwnerOptions.length === 0}
            placeholder={memberOwnerPlaceholder}
            emptyText={t("createKey.userId.noTeamMembers")}
          />
        ) : (
          <div>
            <div className="mb-2 flex">
              <PaginatedSearchSelect
                options={userOptions}
                value={typeof control.value === "string" ? control.value : undefined}
                onValueChange={control.onChange}
                onSearchChange={fetchUsers}
                isLoading={userSearchLoading}
                placeholder={t("createKey.userId.searchPlaceholder")}
                emptyText={t("createKey.userId.noUsers")}
                loadingText={t("createKey.userId.searching")}
                inputId={control.id}
                aria-required={control["aria-required"] === "true" ? true : undefined}
                aria-invalid={control["aria-invalid"] === "true" ? true : undefined}
                aria-describedby={control["aria-describedby"]}
              />
              <Button variant="outline" className="ml-2" onClick={() => setIsCreateUserModalVisible(true)}>
                {t("createKey.userId.createUser")}
              </Button>
            </div>
            <div className="text-xs text-muted-foreground">{t("createKey.userId.searchHelp")}</div>
          </div>
        )
      }
    </MountedFormField>
  );

  return (
    <div>
      {userRole && rolesWithWriteAccess.includes(userRole) && (
        <Button className="mx-auto" onClick={() => setIsModalVisible(true)} data-testid="create-key-button">
          {t("createKey.createButton")}
        </Button>
      )}
      <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-foreground">{t("createKey.dialogTitle")}</DialogTitle>
          </DialogHeader>
          <MountedFormProvider value={mountedForm}>
            <form onSubmit={handleSubmit}>
              {/* Section 1: Key Ownership */}
              <div className="mb-8">
                <h3 className="text-lg font-medium text-foreground mb-4">{t("createKey.sections.ownership")}</h3>
                <Field className="mb-4">
                  <FieldLabel>
                    <span>
                      {t("createKey.owner.label")}{" "}
                      <SimpleTooltip content={t("createKey.owner.tooltip")}>
                        <Info className="ml-1 inline size-3.5 align-text-bottom" />
                      </SimpleTooltip>
                    </span>
                  </FieldLabel>
                  <RadioGroup
                    className="flex flex-wrap items-center gap-4"
                    value={keyOwner}
                    onValueChange={(value: unknown) => changeKeyOwner(String(value))}
                  >
                    <label className={KEY_OWNER_LABEL_CLASS}>
                      <RadioGroupItem value="you" />
                      {t("createKey.owner.you")}
                    </label>
                    <label className={KEY_OWNER_LABEL_CLASS}>
                      <RadioGroupItem value="service_account" />
                      {t("createKey.owner.serviceAccount")}
                    </label>
                    {ownerScope !== "none" && (
                      <label className={KEY_OWNER_LABEL_CLASS}>
                        <RadioGroupItem value="another_user" />
                        {t("createKey.owner.anotherUser")}
                      </label>
                    )}
                    <label className={KEY_OWNER_LABEL_CLASS}>
                      <RadioGroupItem value="agent" />
                      {t("createKey.owner.agent")} <Badge>{t("createKey.owner.newBadge")}</Badge>
                    </label>
                  </RadioGroup>
                </Field>

                {!ownerLimitedToTeamMembers && ownerField}
                {keyOwner === "agent" && (
                  <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-md dark:bg-purple-950 dark:border-purple-800">
                    <div className="mb-3">
                      <label htmlFor="create-key-agent" className="text-sm font-medium text-foreground">
                        {t("createKey.agent.label")} <span className="text-destructive">*</span>
                      </label>
                    </div>
                    <SearchSelect
                      inputId="create-key-agent"
                      placeholder={t("createKey.agent.placeholder")}
                      emptyText={t("createKey.agent.empty")}
                      value={selectedAgentId}
                      onValueChange={setSelectedAgentId}
                      options={agentsList.map((a) => ({
                        label: a.agent_name || a.agent_id,
                        value: a.agent_id,
                      }))}
                    />
                    <div className="text-xs text-muted-foreground mt-2">{t("createKey.agent.help")}</div>
                  </div>
                )}
                <MountedFormField
                  label={
                    <span>
                      {t("createKey.organization.label")}{" "}
                      <SimpleTooltip content={t("createKey.organization.tooltip")}>
                        <Info className="ml-1 inline size-3.5 align-text-bottom" />
                      </SimpleTooltip>
                    </span>
                  }
                  name="organization_id"
                  className="mt-4"
                >
                  {(control) => (
                    <OrganizationDropdown
                      id={control.id}
                      value={typeof control.value === "string" ? control.value : null}
                      organizations={organizations}
                      loading={isOrganizationsLoading}
                      disabled={userRole !== "Admin"}
                      onChange={changeOrganization(control.onChange)}
                    />
                  )}
                </MountedFormField>
                <MountedFormField
                  label={
                    <span>
                      {t("createKey.team.label")}{" "}
                      <SimpleTooltip content={t("createKey.team.tooltip")}>
                        <Info className="ml-1 inline size-3.5 align-text-bottom" />
                      </SimpleTooltip>
                    </span>
                  }
                  name="team_id"
                  className="mt-4"
                  required={ownerRequiresTeam}
                  rules={requiredRule(
                    ownerRequiresTeam,
                    ownerLimitedToTeamMembers
                      ? t("createKey.team.requiredMemberTeam")
                      : t("createKey.team.requiredServiceAccount"),
                  )}
                  help={teamFieldHelp}
                >
                  {(control) => (
                    <TeamDropdown
                      id={control.id}
                      value={typeof control.value === "string" ? control.value : null}
                      onChange={control.onChange}
                      disabled={selectedProjectId !== null}
                      organizationId={selectedOrganizationId}
                      onTeamSelect={selectTeam}
                      filterTeam={ownerLimitedToTeamMembers ? (team) => administersTeam(team, userID) : undefined}
                    />
                  )}
                </MountedFormField>
                {ownerLimitedToTeamMembers && ownerField}
                {enableProjectsUI && (
                  <MountedFormField
                    label={
                      <span>
                        {t("createKey.project.label")}{" "}
                        <SimpleTooltip content={t("createKey.project.tooltip")}>
                          <Info className="ml-1 inline size-3.5 align-text-bottom" />
                        </SimpleTooltip>
                      </span>
                    }
                    name="project_id"
                    className="mt-4"
                  >
                    {(control) => (
                      <ProjectDropdown
                        id={control.id}
                        value={typeof control.value === "string" ? control.value : null}
                        projects={projects}
                        teamId={selectedCreateKeyTeam?.team_id}
                        loading={isProjectsLoading || !teams}
                        onChange={changeProject(control.onChange)}
                      />
                    )}
                  </MountedFormField>
                )}
              </div>

              {/* Show message when team selection is required */}
              {isFormDisabled && (
                <div className="mb-8 p-4 bg-info/10 border border-info/20 rounded-md">
                  <p className="text-info text-sm">{t("createKey.teamRequiredNotice")}</p>
                </div>
              )}

              {/* Section 2: Key Details */}
              {!isFormDisabled && (
                <div className="mb-8">
                  <h3 className="text-lg font-medium text-foreground mb-4">{t("createKey.sections.details")}</h3>
                  <MountedFormField
                    label={
                      <span>
                        {keyOwner === "you" || keyOwner === "another_user"
                          ? t("createKey.keyName.label")
                          : t("createKey.keyName.serviceAccountLabel")}{" "}
                        <SimpleTooltip
                          content={
                            keyOwner === "you" || keyOwner === "another_user"
                              ? t("createKey.keyName.tooltip")
                              : t("createKey.keyName.serviceAccountTooltip")
                          }
                        >
                          <Info className="ml-1 inline size-3.5 align-text-bottom" />
                        </SimpleTooltip>
                      </span>
                    }
                    name="key_alias"
                    required
                    rules={requiredRule(
                      true,
                      t("createKey.keyName.required", {
                        name:
                          keyOwner === "you"
                            ? t("createKey.keyName.requiredNames.keyName")
                            : t("createKey.keyName.requiredNames.serviceAccountId"),
                      }),
                    )}
                    help={t("createKey.keyName.help")}
                  >
                    {(control) => <Input {...control} value={(control.value as string | undefined) ?? ""} />}
                  </MountedFormField>

                  <MountedFormField
                    label={
                      <span>
                        {t("createKey.models.label")}{" "}
                        <SimpleTooltip content={t("createKey.models.tooltip")}>
                          <Info className="ml-1 inline size-3.5 align-text-bottom" />
                        </SimpleTooltip>
                      </span>
                    }
                    name="models"
                    help={
                      keyType === "management" || keyType === "read_only"
                        ? t("createKey.models.helpDisabled")
                        : t("createKey.models.helpOptional")
                    }
                    className="mt-4"
                  >
                    {(control) => (
                      <MultiSelect
                        id={control.id}
                        options={modelOptions}
                        value={(control.value as string[] | undefined) ?? []}
                        placeholder={t("createKey.models.placeholder")}
                        disabled={keyType === "management" || keyType === "read_only"}
                        onValueChange={(values) => {
                          control.onChange(values);
                          if (values.includes("all-team-models")) {
                            form.setValue("models", ["all-team-models"]);
                          } else if (values.includes("all-proxy-models")) {
                            form.setValue("models", ["all-proxy-models"]);
                          }
                        }}
                      />
                    )}
                  </MountedFormField>

                  <MountedFormField
                    label={
                      <span>
                        {t("createKey.keyType.label")}{" "}
                        <SimpleTooltip content={t("createKey.keyType.tooltip")}>
                          <Info className="ml-1 inline size-3.5 align-text-bottom" />
                        </SimpleTooltip>
                      </span>
                    }
                    name="key_type"
                    className="mt-4"
                  >
                    {(control) => (
                      <Select
                        items={keyTypeOptions}
                        value={control.value as string | undefined}
                        onValueChange={(value: string | null) =>
                          value != null && changeKeyType(control.onChange)(value)
                        }
                      >
                        <SelectTrigger
                          id={control.id}
                          className="w-full"
                          aria-invalid={control["aria-invalid"]}
                          aria-describedby={control["aria-describedby"]}
                        >
                          <SelectValue placeholder={t("createKey.keyType.placeholder")} />
                        </SelectTrigger>
                        <SelectContent>
                          {keyTypeOptions.map((option) => (
                            <SelectItem key={option.value} value={option.value}>
                              <div className="py-1">
                                <div className="font-medium">{option.label}</div>
                                <div className="mt-0.5 text-[11px] text-muted-foreground">{option.hint}</div>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </MountedFormField>
                </div>
              )}

              {/* Section 3: Optional Settings */}
              {!isFormDisabled && (
                <div className="mb-8">
                  <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                    <h3 className="m-0 text-lg font-medium text-foreground">
                      <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                        {t("createKey.sections.optional")}
                        <ChevronDown className={SECTION_CHEVRON_CLASS} />
                      </CollapsibleTrigger>
                    </h3>
                    <CollapsibleContent className="px-4 pb-3">
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.maxBudget.label")}{" "}
                            <SimpleTooltip content={t("createKey.maxBudget.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="max_budget"
                        help={t("createKey.maxBudget.help", {
                          budget:
                            team?.max_budget !== null && team?.max_budget !== undefined
                              ? team?.max_budget
                              : t("createKey.limits.unlimited"),
                        })}
                        rules={ceilingRule(team?.max_budget, (limit) =>
                          t("createKey.maxBudget.ceiling", { limit: formatNumberWithCommas(limit, 4) }),
                        )}
                      >
                        {(control) => (
                          <NumericalInput
                            {...control}
                            value={control.value as number | string | undefined}
                            step={0.01}
                            precision={2}
                            width={200}
                          />
                        )}
                      </MountedFormField>
                      <div className="mt-4">
                        <MountedBudgetAlertEmailsEditor
                          control={form.control}
                          label={
                            <span>
                              {t("keyDetail.maxBudgetAlertEmails.budgetAlertThresholds")}{" "}
                              <SimpleTooltip content={t("keyDetail.maxBudgetAlertEmails.budgetAlertThresholdsHint")}>
                                <Info className="ml-1 inline size-3.5 align-text-bottom" />
                              </SimpleTooltip>
                            </span>
                          }
                          text={{
                            thresholdPlaceholder: t("keyDetail.maxBudgetAlertEmails.thresholdPlaceholder"),
                            recipientsPlaceholder: t("keyDetail.maxBudgetAlertEmails.recipientsPlaceholder"),
                            removeThreshold: t("keyDetail.maxBudgetAlertEmails.removeThreshold"),
                            addThreshold: t("keyDetail.maxBudgetAlertEmails.addThreshold"),
                          }}
                        />
                      </div>
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.resetBudget.label")}{" "}
                            <SimpleTooltip content={t("createKey.resetBudget.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="budget_duration"
                        help={t("createKey.resetBudget.help", {
                          duration:
                            team?.budget_duration !== null && team?.budget_duration !== undefined
                              ? team?.budget_duration
                              : t("createKey.resetBudget.none"),
                        })}
                      >
                        {(control) => (
                          <BudgetDurationDropdown
                            id={control.id}
                            value={control.value as string | null | undefined}
                            showNeverResets
                            placeholder={t("createKey.resetBudget.placeholder")}
                            onChange={(next) => control.onChange(next ?? undefined)}
                          />
                        )}
                      </MountedFormField>
                      <Field className="mt-4">
                        <FieldLabel>
                          <span>
                            {t("createKey.budgetWindows.label")}{" "}
                            <SimpleTooltip content={t("createKey.budgetWindows.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        </FieldLabel>
                        <BudgetWindowsEditor value={budgetLimits} onChange={setBudgetLimits} />
                      </Field>
                      <Field className="mt-4">
                        <FieldLabel>
                          <span>
                            {t("createKey.perModelBudgets.label")}{" "}
                            <SimpleTooltip content={t("createKey.perModelBudgets.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        </FieldLabel>
                        <ModelMaxBudgetEditor
                          value={modelMaxBudget}
                          onChange={setModelMaxBudget}
                          availableModels={modelsToPick}
                          premiumUser={premiumUser === true}
                        />
                      </Field>
                      <Field className="mt-4">
                        <FieldLabel>
                          <span>
                            {t("createKey.budgetFallbacks.label")}{" "}
                            <SimpleTooltip content={t("createKey.budgetFallbacks.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        </FieldLabel>
                        <BudgetFallbacksEditor
                          key={budgetFallbacksKey}
                          value={budgetFallbacks}
                          onChange={setBudgetFallbacks}
                          availableModels={modelsToPick}
                        />
                      </Field>
                      {keyOwner === "service_account" && isProxyAdminRole(userRole ?? "") && (
                        <MountedFormField
                          className="mt-4"
                          label={
                            <span>
                              {t("createKey.defaultCustomerBudget.label")}{" "}
                              <SimpleTooltip content={t(END_USER_BUDGET_HINT)}>
                                <Info className="ml-1 inline size-3.5 align-text-bottom" />
                              </SimpleTooltip>
                            </span>
                          }
                          name="end_user_budget_id"
                        >
                          {(control) => (
                            <EndUserBudgetSelect
                              id={control.id}
                              accessToken={accessToken}
                              value={typeof control.value === "string" ? control.value : null}
                              onChange={control.onChange}
                              canEdit
                            />
                          )}
                        </MountedFormField>
                      )}
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.tpm.label")}{" "}
                            <SimpleTooltip content={t("createKey.tpm.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="tpm_limit"
                        help={t("createKey.tpm.help", {
                          limit:
                            team?.tpm_limit !== null && team?.tpm_limit !== undefined
                              ? team?.tpm_limit
                              : t("createKey.limits.unlimited"),
                        })}
                        rules={ceilingRule(team?.tpm_limit, (limit) => t("createKey.tpm.ceiling", { limit }))}
                      >
                        {(control) => (
                          <NumericalInput
                            {...control}
                            value={control.value as number | string | undefined}
                            step={1}
                            width={400}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField name="tpm_limit_type" bare>
                        {(control) => (
                          <RateLimitTypeFormItem
                            type="tpm"
                            name="tpm_limit_type"
                            className="mt-4"
                            showDetailedDescriptions
                            id={control.id}
                            value={control.value as string | null | undefined}
                            onChange={control.onChange}
                            aria-invalid={control["aria-invalid"] ? true : undefined}
                            aria-describedby={control["aria-describedby"]}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.rpm.label")}{" "}
                            <SimpleTooltip content={t("createKey.rpm.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="rpm_limit"
                        help={t("createKey.rpm.help", {
                          limit:
                            team?.rpm_limit !== null && team?.rpm_limit !== undefined
                              ? team?.rpm_limit
                              : t("createKey.limits.unlimited"),
                        })}
                        rules={ceilingRule(team?.rpm_limit, (limit) => t("createKey.rpm.ceiling", { limit }))}
                      >
                        {(control) => (
                          <NumericalInput
                            {...control}
                            value={control.value as number | string | undefined}
                            step={1}
                            width={400}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField name="rpm_limit_type" bare>
                        {(control) => (
                          <RateLimitTypeFormItem
                            type="rpm"
                            name="rpm_limit_type"
                            className="mt-4"
                            showDetailedDescriptions
                            id={control.id}
                            value={control.value as string | null | undefined}
                            onChange={control.onChange}
                            aria-invalid={control["aria-invalid"] ? true : undefined}
                            aria-describedby={control["aria-describedby"]}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.tpd.label")}{" "}
                            <SimpleTooltip content={t("createKey.tpd.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="tpd_limit"
                        help={t("createKey.tpd.help", {
                          limit:
                            team?.tpd_limit !== null && team?.tpd_limit !== undefined
                              ? team?.tpd_limit
                              : t("createKey.limits.unlimited"),
                        })}
                        rules={ceilingRule(team?.tpd_limit, (limit) => t("createKey.tpd.ceiling", { limit }))}
                      >
                        {(control) => (
                          <NumericalInput
                            {...control}
                            value={control.value as number | string | undefined}
                            step={1}
                            width={400}
                          />
                        )}
                      </MountedFormField>
                      <Field className="mt-4">
                        <FieldLabel>
                          <span>
                            {t("createKey.tagRateLimits.label")}{" "}
                            <SimpleTooltip content={t("createKey.tagRateLimits.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        </FieldLabel>
                        <TagRateLimitEditor value={tagRateLimits} onChange={setTagRateLimits} />
                      </Field>
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.throttle.label")}{" "}
                            <SimpleTooltip content={t("createKey.throttle.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="throttle_on_budget_exceeded"
                      >
                        {(control) => (
                          <Switch
                            id={control.id}
                            checked={control.value === true}
                            onCheckedChange={control.onChange}
                            aria-describedby={control["aria-describedby"]}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        className="mt-4"
                        label={
                          <span>
                            {t("createKey.promptCaching.label")}{" "}
                            <SimpleTooltip content={t("createKey.promptCaching.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="enable_prompt_caching"
                      >
                        {(control) => (
                          <Switch
                            id={control.id}
                            checked={control.value === true}
                            onCheckedChange={control.onChange}
                            aria-describedby={control["aria-describedby"]}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.guardrails.label")}{" "}
                            <SimpleTooltip content={t("createKey.guardrails.tooltip")}>
                              <a
                                href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                              >
                                <Info className="ml-1 inline size-3.5 align-text-bottom" />
                              </a>
                            </SimpleTooltip>
                          </span>
                        }
                        name="guardrails"
                        className="mt-4"
                        help={canEditGuardrails ? t("createKey.guardrails.help") : t("createKey.guardrails.premium")}
                      >
                        {(control) => (
                          <TagsInput
                            id={control.id}
                            value={(control.value as string[] | undefined) ?? []}
                            onValueChange={control.onChange}
                            disabled={!canEditGuardrails}
                            placeholder={
                              !canEditGuardrails
                                ? t("createKey.guardrails.premium")
                                : t("createKey.guardrails.placeholder")
                            }
                            options={guardrailsList.map((name) => ({ value: name, label: name }))}
                          />
                        )}
                      </MountedFormField>
                      {userRole != null && isProxyAdminRole(userRole) && (
                        <MountedFormField
                          label={
                            <span>
                              {t("createKey.disableGlobalGuardrails.label")}{" "}
                              <SimpleTooltip content={t("createKey.disableGlobalGuardrails.tooltip")}>
                                <a
                                  href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                                >
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </a>
                              </SimpleTooltip>
                            </span>
                          }
                          name="disable_global_guardrails"
                          className="mt-4"
                          help={
                            canEditGuardrails
                              ? t("createKey.disableGlobalGuardrails.help")
                              : t("createKey.disableGlobalGuardrails.premium")
                          }
                        >
                          {(control) => (
                            <Switch
                              id={control.id}
                              checked={control.value === true}
                              onCheckedChange={control.onChange}
                              disabled={!canEditGuardrails}
                              aria-describedby={control["aria-describedby"]}
                            />
                          )}
                        </MountedFormField>
                      )}
                      {canViewPolicies && (
                        <MountedFormField
                          label={
                            <span>
                              {t("createKey.policies.label")}{" "}
                              <SimpleTooltip content={t("createKey.policies.tooltip")}>
                                <a
                                  href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                                >
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </a>
                              </SimpleTooltip>
                            </span>
                          }
                          name="policies"
                          className="mt-4"
                          help={premiumUser ? t("createKey.policies.help") : t("createKey.policies.premium")}
                        >
                          {(control) => (
                            <TagsInput
                              id={control.id}
                              value={(control.value as string[] | undefined) ?? []}
                              onValueChange={control.onChange}
                              disabled={!premiumUser}
                              placeholder={
                                !premiumUser ? t("createKey.policies.premium") : t("createKey.policies.placeholder")
                              }
                              options={policiesList.map((name) => ({ value: name, label: name }))}
                            />
                          )}
                        </MountedFormField>
                      )}
                      {canViewPrompts && (
                        <MountedFormField
                          label={
                            <span>
                              {t("createKey.prompts.label")}{" "}
                              <SimpleTooltip content={t("createKey.prompts.tooltip")}>
                                <a
                                  href="https://docs.litellm.ai/docs/proxy/prompt_management"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                                >
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </a>
                              </SimpleTooltip>
                            </span>
                          }
                          name="prompts"
                          className="mt-4"
                          help={premiumUser ? t("createKey.prompts.help") : t("createKey.prompts.premium")}
                        >
                          {(control) => (
                            <TagsInput
                              id={control.id}
                              value={(control.value as string[] | undefined) ?? []}
                              onValueChange={control.onChange}
                              disabled={!premiumUser}
                              placeholder={
                                !premiumUser ? t("createKey.prompts.premium") : t("createKey.prompts.placeholder")
                              }
                              options={promptsList.map((name) => ({ value: name, label: name }))}
                            />
                          )}
                        </MountedFormField>
                      )}
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.accessGroups.label")}{" "}
                            <SimpleTooltip content={t("createKey.accessGroups.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="access_group_ids"
                        className="mt-4"
                        help={t("createKey.accessGroups.help")}
                      >
                        {(control) => (
                          <AccessGroupSelector
                            value={control.value as string[] | undefined}
                            onChange={control.onChange}
                            placeholder={t("createKey.accessGroups.placeholder")}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.passThroughRoutes.label")}{" "}
                            <SimpleTooltip content={t("createKey.passThroughRoutes.tooltip")}>
                              <a
                                href="https://docs.litellm.ai/docs/proxy/pass_through"
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                              >
                                <Info className="ml-1 inline size-3.5 align-text-bottom" />
                              </a>
                            </SimpleTooltip>
                          </span>
                        }
                        name="allowed_passthrough_routes"
                        className="mt-4"
                        help={
                          premiumUser ? t("createKey.passThroughRoutes.help") : t("createKey.passThroughRoutes.premium")
                        }
                      >
                        {(control) => (
                          <PassThroughRoutesSelector
                            value={control.value as string[] | undefined}
                            onChange={control.onChange}
                            accessToken={accessToken}
                            placeholder={
                              !premiumUser
                                ? t("createKey.passThroughRoutes.premium")
                                : t("createKey.passThroughRoutes.placeholder")
                            }
                            disabled={!premiumUser}
                            teamId={selectedCreateKeyTeam ? selectedCreateKeyTeam.team_id : null}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.vectorStores.label")}{" "}
                            <SimpleTooltip content={t("createKey.vectorStores.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="allowed_vector_store_ids"
                        className="mt-4"
                        help={t("createKey.vectorStores.help")}
                      >
                        {(control) => (
                          <VectorStoreSelector
                            onChange={control.onChange}
                            value={control.value as string[] | undefined}
                            accessToken={accessToken}
                            placeholder={t("createKey.vectorStores.placeholder")}
                          />
                        )}
                      </MountedFormField>
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.metadata.label")}{" "}
                            <SimpleTooltip content={t("createKey.metadata.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="metadata"
                        className="mt-4"
                      >
                        {(control) => (
                          <>
                            <StorePromptsMetadataToggle
                              value={control.value as string | undefined}
                              onChange={control.onChange}
                            />
                            <Textarea
                              {...control}
                              value={(control.value as string | undefined) ?? ""}
                              rows={4}
                              placeholder={t("createKey.metadata.placeholder")}
                              onBlur={() => {
                                control.onBlur();
                                moveMetadataBudgetAlertsToRows(
                                  form.getValues("metadata") as string | undefined,
                                  form.getValues(KEY_MAX_BUDGET_ALERT_EMAILS_KEY) as BudgetAlertRow[] | undefined,
                                  (metadata, rows) => {
                                    form.setValue("metadata", metadata, { shouldDirty: true });
                                    form.setValue(KEY_MAX_BUDGET_ALERT_EMAILS_KEY, rows, { shouldDirty: true });
                                  },
                                  t,
                                );
                              }}
                            />
                          </>
                        )}
                      </MountedFormField>
                      <MountedFormField
                        label={
                          <span>
                            {t("createKey.tags.label")}{" "}
                            <SimpleTooltip content={t("createKey.tags.tooltip")}>
                              <Info className="ml-1 inline size-3.5 align-text-bottom" />
                            </SimpleTooltip>
                          </span>
                        }
                        name="tags"
                        className="mt-4"
                        help={t("createKey.tags.help")}
                      >
                        {(control) => (
                          <TagsInput
                            id={control.id}
                            value={(control.value as string[] | undefined) ?? []}
                            onValueChange={control.onChange}
                            placeholder={t("createKey.tags.placeholder")}
                            tokenSeparators={[","]}
                            options={tagOptions}
                          />
                        )}
                      </MountedFormField>
                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.mcp")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <MountedFormField
                            label={
                              <span>
                                {t("createKey.mcpServers.label")}{" "}
                                <SimpleTooltip content={t("createKey.mcpServers.tooltip")}>
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </SimpleTooltip>
                              </span>
                            }
                            name="allowed_mcp_servers_and_groups"
                            help={t("createKey.mcpServers.help")}
                          >
                            {(control) => (
                              <MCPServerSelector
                                onChange={control.onChange}
                                value={control.value as McpSelectorValue | undefined}
                                accessToken={accessToken}
                                teamId={selectedCreateKeyTeam?.team_id ?? null}
                                userId={typeof createKeyOwner === "string" ? createKeyOwner : null}
                                placeholder={t("createKey.mcpServers.placeholder")}
                                allowNoMcpServers
                              />
                            )}
                          </MountedFormField>

                          {/* Hidden field to register mcp_tool_permissions with the form */}
                          <MountedFormField name="mcp_tool_permissions" bare>
                            {(control) => <input type="hidden" id={control.id} name={control.name} />}
                          </MountedFormField>

                          <McpToolPermissionsField
                            accessToken={accessToken}
                            control={form.control}
                            setValue={form.setValue}
                          />
                        </CollapsibleContent>
                      </Collapsible>

                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.agent")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <MountedFormField
                            label={
                              <span>
                                {t("createKey.agents.label")}{" "}
                                <SimpleTooltip content={t("createKey.agents.tooltip")}>
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </SimpleTooltip>
                              </span>
                            }
                            name="allowed_agents_and_groups"
                            help={t("createKey.agents.help")}
                          >
                            {(control) => (
                              <AgentSelector
                                onChange={control.onChange}
                                value={control.value as AgentSelectorValue | undefined}
                                accessToken={accessToken}
                                placeholder={t("createKey.agents.placeholder")}
                              />
                            )}
                          </MountedFormField>
                        </CollapsibleContent>
                      </Collapsible>

                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.skill")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <MountedFormField
                            label={
                              <span>
                                {t("createKey.skills.label")}{" "}
                                <SimpleTooltip content={t("createKey.skills.tooltip")}>
                                  <Info className="ml-1 inline size-3.5 align-text-bottom" />
                                </SimpleTooltip>
                              </span>
                            }
                            name="allowed_skills"
                            help={t("createKey.skills.help")}
                          >
                            {(control) => (
                              <SkillSelector
                                onChange={control.onChange}
                                value={control.value as string[] | undefined}
                                accessToken={accessToken}
                                placeholder={t("createKey.skills.placeholder")}
                              />
                            )}
                          </MountedFormField>
                        </CollapsibleContent>
                      </Collapsible>

                      {premiumUser ? (
                        <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                          <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                            <b>{t("createKey.sections.logging")}</b>
                            <ChevronDown className={SECTION_CHEVRON_CLASS} />
                          </CollapsibleTrigger>
                          <CollapsibleContent className="px-4 pb-3">
                            <div className="mt-4">
                              <PremiumLoggingSettings
                                value={loggingSettings}
                                onChange={setLoggingSettings}
                                premiumUser={true}
                                disabledCallbacks={disabledCallbacks}
                                onDisabledCallbacksChange={setDisabledCallbacks}
                              />
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      ) : (
                        <SimpleTooltip
                          className="w-full"
                          content={
                            <span>
                              {t("createKey.logging.enterpriseTooltip")}
                              <a href="https://www.litellm.ai/enterprise" target="_blank">
                                https://www.litellm.ai/enterprise
                              </a>
                            </span>
                          }
                          side="top"
                        >
                          <div style={{ position: "relative" }}>
                            <div style={{ opacity: 0.5 }}>
                              <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                                <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                                  <b>{t("createKey.sections.logging")}</b>
                                  <ChevronDown className={SECTION_CHEVRON_CLASS} />
                                </CollapsibleTrigger>
                                <CollapsibleContent className="px-4 pb-3">
                                  <div className="mt-4">
                                    <PremiumLoggingSettings
                                      value={loggingSettings}
                                      onChange={setLoggingSettings}
                                      premiumUser={false}
                                      disabledCallbacks={disabledCallbacks}
                                      onDisabledCallbacksChange={setDisabledCallbacks}
                                    />
                                  </div>
                                </CollapsibleContent>
                              </Collapsible>
                            </div>
                            <div style={{ position: "absolute", inset: 0, cursor: "not-allowed" }} />
                          </div>
                        </SimpleTooltip>
                      )}

                      <Collapsible
                        key={`router-settings-accordion-${routerSettingsKey}`}
                        className="mt-4 mb-4 overflow-hidden rounded-lg border"
                      >
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.router")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <div className="mt-4 w-full">
                            <RouterSettingsAccordion
                              key={routerSettingsKey}
                              ref={routerSettingsRef}
                              accessToken={accessToken || ""}
                              teamId={selectedCreateKeyTeam?.team_id}
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

                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.modelAliases")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <div className="mt-4">
                            <p className="text-sm text-muted-foreground mb-4">
                              {t("createKey.modelAliases.description")}
                            </p>
                            <ModelAliasManager
                              accessToken={accessToken}
                              initialModelAliases={modelAliases}
                              onAliasUpdate={setModelAliases}
                              showExampleConfig={false}
                            />
                          </div>
                        </CollapsibleContent>
                      </Collapsible>

                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <b>{t("createKey.sections.lifecycle")}</b>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <div className="mt-4">
                            <MountedFormField name="duration" bare>
                              {(control) => (
                                <KeyLifecycleSettings
                                  id={control.id}
                                  value={control.value as string | undefined}
                                  onChange={control.onChange}
                                  autoRotationEnabled={autoRotationEnabled}
                                  onAutoRotationChange={setAutoRotationEnabled}
                                  rotationInterval={rotationInterval}
                                  onRotationIntervalChange={setRotationInterval}
                                  isCreateMode={true}
                                />
                              )}
                            </MountedFormField>
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                      <Collapsible className="mt-4 mb-4 overflow-hidden rounded-lg border">
                        <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
                          <div className="flex items-center gap-2">
                            <b>{t("createKey.sections.advanced")}</b>
                            <SimpleTooltip
                              content={
                                <span>
                                  <Trans
                                    i18nKey="createKey.advanced.tooltip"
                                    components={{
                                      docs: (
                                        <a
                                          href={
                                            proxyBaseUrl
                                              ? `${proxyBaseUrl}/#/key%20management/generate_key_fn_key_generate_post`
                                              : `/#/key%20management/generate_key_fn_key_generate_post`
                                          }
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-info hover:text-info/80"
                                        />
                                      ),
                                    }}
                                  />
                                </span>
                              }
                            >
                              <Info className="size-4 text-muted-foreground hover:text-foreground cursor-help" />
                            </SimpleTooltip>
                          </div>
                          <ChevronDown className={SECTION_CHEVRON_CLASS} />
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-4 pb-3">
                          <SchemaFormFields
                            schemaComponent="GenerateKeyRequest"
                            setValue={form.setValue}
                            excludedFields={
                              disableCustomApiKeys
                                ? GENERATE_KEY_EXCLUDED_FIELDS_WITH_KEY
                                : GENERATE_KEY_EXCLUDED_FIELDS
                            }
                          />
                        </CollapsibleContent>
                      </Collapsible>
                    </CollapsibleContent>
                  </Collapsible>
                </div>
              )}

              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button type="submit" disabled={isFormDisabled}>
                  {t("createKey.submit")}
                </Button>
              </div>
            </form>
          </MountedFormProvider>
        </DialogContent>
      </Dialog>

      {/* Add the Create User Modal */}
      {isCreateUserModalVisible && (
        <Dialog open={isCreateUserModalVisible} onOpenChange={(open) => !open && setIsCreateUserModalVisible(false)}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
            <DialogHeader>
              <DialogTitle>{t("createKey.createUserDialog.title")}</DialogTitle>
            </DialogHeader>
            <CreateUserButton
              userID={userID ?? ""}
              accessToken={accessToken}
              possibleUIRoles={possibleUIRoles}
              onUserCreated={handleUserCreated}
              isEmbedded={true}
            />
          </DialogContent>
        </Dialog>
      )}

      {apiKey && (
        <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
            <div className="grid grid-cols-1 gap-2 w-full">
              <DialogTitle className="text-lg font-medium text-foreground">{t("createKey.created.title")}</DialogTitle>
              {apiKey != null ? (
                <CreatedKeyDisplay apiKey={apiKey} />
              ) : (
                <p className="text-sm">{t("createKey.created.creating")}</p>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default CreateKey;
