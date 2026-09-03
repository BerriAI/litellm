import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import PolicySelector from "@/components/policies/PolicySelector";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { TooltipProvider } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import React, { useEffect, useRef, useState } from "react";
import { hasCapability } from "../../utils/capabilities";
import { isProxyAdminRole, rolesWithWriteAccess } from "../../utils/roles";
import AgentSelector from "../agent_management/AgentSelector";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { mapInternalToDisplayNames } from "../callback_info_helpers";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import OrganizationDropdown from "../common_components/OrganizationDropdown";
import RouterSettingsAccordion, { RouterSettingsAccordionRef } from "../common_components/RouterSettingsAccordion";
import { routerSettingsEditorValue, routerSettingsUpdate } from "../common_components/routerSettingsPayload";
import { estimateTooltips, withNormalizedEstimates } from "./estimatedOutputTokens";
import {
  currentValuePlaceholder,
  keyTypeFromRoutes,
  modelSentinelOptions,
  parseAllowedRoutes,
} from "./keyEditFieldNormalizers";
import { KeyTypeSelect, labelWithHint } from "./KeyEditViewControls";
import {
  AgentsAndGroups,
  KeyEditFormValues,
  keyEditFormSchema,
  McpServersAndGroups,
  toKeyEditFormValues,
  toSubmittedValues,
} from "./keyEditFormValues";
import { BudgetFallbacksEditor } from "../key_team_helpers/BudgetFallbacksEditor";
import { ModelMaxBudgetField } from "../key_team_helpers/ModelMaxBudgetEditor";
import { useModelMaxBudgetField } from "../key_team_helpers/useModelMaxBudgetField";
import { BudgetWindowEntry, BudgetWindowsEditor } from "../key_team_helpers/BudgetWindowsEditor";
import {
  TagRateLimitEditor,
  TagRateLimitEntry,
  tagLimitsToRows,
  tagRowsToLimits,
} from "../key_team_helpers/TagRateLimitEditor";
import { excludeProxyWideSentinel, hasAllModelsSentinel } from "../key_team_helpers/fetch_available_models_team_key";
import { KeyResponse } from "../key_team_helpers/key_list";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import { toast } from "@/lib/toast";
import { getPromptsList, modelAvailableCall, tagListCall } from "../networking";
import { fetchTeamModels } from "../organisms/create_key_button";
import NumericalInput from "../shared/numerical_input";
import { MultiSelect } from "../shared/MultiSelect";
import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { Tag } from "../tag_management/types";
import EditLoggingSettings from "../team/EditLoggingSettings";
import { useZodForm } from "@/lib/forms/useZodForm";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";

interface KeyEditViewProps {
  keyData: KeyResponse;
  onCancel: () => void;
  onSubmit: (values: any) => Promise<void>;
  teams?: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  premiumUser?: boolean;
}

export function KeyEditView({
  keyData,
  onCancel,
  onSubmit,
  teams,
  accessToken,
  userID,
  userRole,
  premiumUser = false,
}: KeyEditViewProps) {
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const canViewPolicies = hasCapability(userRole, "viewPolicies");
  const canViewPrompts = hasCapability(userRole, "viewPrompts");
  const canEditEstimates = userRole != null && isProxyAdminRole(userRole);
  const estimateTooltip = estimateTooltips(canEditEstimates);
  const form = useZodForm<KeyEditFormValues, KeyEditFormValues>(keyEditFormSchema, {
    defaultValues: toKeyEditFormValues(keyData),
  });
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  const team = teams?.find((team) => team.team_id === keyData.team_id);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>(
    Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
      ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
      : [],
  );
  const [selectedOrganizationId, setSelectedOrganizationId] = useState<string | null>(keyData.organization_id || null);
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(keyData.auto_rotate || false);
  const [rotationInterval, setRotationInterval] = useState<string>(keyData.rotation_interval || "");
  const [neverExpire, setNeverExpire] = useState<boolean>(!keyData.expires);
  const [isKeySaving, setIsKeySaving] = useState(false);
  const [budgetLimits, setBudgetLimits] = useState<BudgetWindowEntry[]>(
    Array.isArray(keyData.budget_limits) ? keyData.budget_limits : [],
  );
  const [tagRateLimits, setTagRateLimits] = useState<TagRateLimitEntry[]>(
    tagLimitsToRows(keyData.metadata?.tag_rpm_limit),
  );
  const [budgetFallbacks, setBudgetFallbacks] = useState<Record<string, string[]>>(
    keyData.budget_fallbacks && typeof keyData.budget_fallbacks === "object" ? keyData.budget_fallbacks : {},
  );
  const modelBudget = useModelMaxBudgetField(keyData.token, keyData.model_max_budget);
  const routerSettingsRef = useRef<RouterSettingsAccordionRef>(null);
  const keyTypeFieldId = React.useId();
  const projectFieldId = React.useId();
  const { data: organizations, isLoading: isOrganizationsLoading } = useOrganizations();
  const { data: projects } = useProjects();
  const { data: uiSettingsData } = useUISettings();
  const enableProjectsUI = Boolean(uiSettingsData?.values?.enable_projects_ui);
  const hasProject = Boolean(keyData.project_id);
  const projectDisplay = (() => {
    if (!keyData.project_id) return null;
    const project = projects?.find((p) => p.project_id === keyData.project_id);
    return project?.project_alias ? `${project.project_alias} (${keyData.project_id})` : keyData.project_id;
  })();

  const allowedRoutesValue = form.watch("allowed_routes");
  const selectedModels = (form.watch("models") as string[] | undefined) ?? [];
  const allowedRoutes = parseAllowedRoutes(allowedRoutesValue);
  const isModelsDisabled = allowedRoutes.includes("management_routes") || allowedRoutes.includes("info_routes");
  const mcpServersAndGroups = form.watch("mcp_servers_and_groups");
  const mcpToolPermissions = form.watch("mcp_tool_permissions");

  useEffect(() => {
    const fetchModels = async () => {
      if (!userID || !userRole || !accessToken) return;

      try {
        if (keyData.team_id === null) {
          // Fetch user models if no team
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          const available_model_names = model_available["data"].map((element: { id: string }) => element.id);
          setAvailableModels(excludeProxyWideSentinel(available_model_names));
        } else if (team?.team_id) {
          // Fetch team models if team exists
          const models = await fetchTeamModels(userID, userRole, accessToken, team.team_id);
          setAvailableModels(excludeProxyWideSentinel(Array.from(new Set([...team.models, ...models]))));
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    const fetchPrompts = async () => {
      if (!accessToken) return;
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(response.prompts.map((prompt) => prompt.prompt_id));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    if (canViewPrompts) fetchPrompts();
    fetchModels();
  }, [userID, userRole, accessToken, team, keyData.team_id, canViewPrompts]);

  // Sync disabled callbacks with form when component mounts
  useEffect(() => {
    form.setValue("disabled_callbacks", disabledCallbacks);
  }, [form, disabledCallbacks]);

  useEffect(() => {
    form.reset(toKeyEditFormValues(keyData));
  }, [keyData, form]);

  // Sync auto-rotation state with form values
  useEffect(() => {
    form.setValue("auto_rotate", autoRotationEnabled);
  }, [autoRotationEnabled, form]);

  useEffect(() => {
    if (rotationInterval) {
      form.setValue("rotation_interval", rotationInterval);
    }
  }, [rotationInterval, form]);

  // Fetch tags for selector
  useEffect(() => {
    const fetchTags = async () => {
      if (!accessToken) return;
      try {
        const response = await tagListCall(accessToken);
        setTagsList(response);
      } catch (error) {
        toast.fromError("Error fetching tags: " + error);
      }
    };
    fetchTags();
  }, [accessToken]);

  const handleSubmit = async (values: any) => {
    try {
      setIsKeySaving(true);

      // Parse allowed_routes from comma-separated string to array
      if (typeof values.allowed_routes === "string") {
        const trimmedInput = values.allowed_routes.trim();
        if (trimmedInput === "") {
          values.allowed_routes = [];
        } else {
          values.allowed_routes = trimmedInput
            .split(",")
            .map((route: string) => route.trim())
            .filter((route: string) => route.length > 0);
        }
      }
      // If it's already an array (shouldn't happen, but handle it), keep as is

      // Backend rejects non-empty allowed_routes from non-admins, so re-sending
      // an unchanged value 403s a team admin. Set compare tolerates reorder.
      const originalRoutesSet = new Set<string>(Array.isArray(keyData.allowed_routes) ? keyData.allowed_routes : []);
      const submittedRoutesSet = new Set<string>(Array.isArray(values.allowed_routes) ? values.allowed_routes : []);
      const allowedRoutesUnchanged =
        originalRoutesSet.size === submittedRoutesSet.size &&
        [...submittedRoutesSet].every((r) => originalRoutesSet.has(r));
      if (allowedRoutesUnchanged) {
        delete values.allowed_routes;
      }

      if (neverExpire) {
        values.duration = null;
      }

      if (keyData.budget_duration && !values.budget_duration) {
        values.budget_duration = null;
      }

      // Reconcile multi-window budget limits from the editor state, dropping
      // incomplete entries (no max_budget). The backend treats any budget_limits
      // in a /key/update request as an admin-only budget change, so re-sending
      // the stored windows on an unrelated edit 403s a non-admin key owner
      // (issue #33246). Only send the field when the user actually changed the
      // windows, mirroring how allowed_routes is dropped above when unchanged:
      // compare on (duration, cap), ignoring server-owned reset_at and order.
      // Sending [] clears every window, so send it only when the user removed
      // the last one; otherwise leave the field off (JSON.stringify drops the
      // undefined key) so an unchanged or incomplete editor state never touches
      // storage.
      const windowSignature = (windows: Array<{ budget_duration: string; max_budget: number | null }> | undefined) =>
        (windows ?? [])
          .filter((w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined)
          .map((w) => `${w.budget_duration}:${w.max_budget}`)
          .sort()
          .join("|");
      const validWindows = budgetLimits.filter(
        (w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined,
      );
      const budgetLimitsUnchanged = windowSignature(keyData.budget_limits) === windowSignature(validWindows);
      if (budgetLimitsUnchanged) {
        // no-op: leave budget_limits off the payload
      } else if (validWindows.length > 0) {
        values.budget_limits = validWindows;
      } else if (budgetLimits.length === 0) {
        values.budget_limits = [];
      }

      // Always send the current per-tag limit map so removing every row
      // clears the stored limits ({} overwrites the metadata field).
      const { tag_rpm_limit } = tagRowsToLimits(tagRateLimits);
      values.tag_rpm_limit = tag_rpm_limit;

      const hadExistingFallbacks = keyData.budget_fallbacks != null && Object.keys(keyData.budget_fallbacks).length > 0;
      if (Object.keys(budgetFallbacks).length > 0) {
        values.budget_fallbacks = budgetFallbacks;
      } else if (hadExistingFallbacks) {
        values.budget_fallbacks = {};
      }

      modelBudget.applyTo(values);

      const routerSettings = routerSettingsUpdate(
        routerSettingsRef.current?.getValue()?.router_settings,
        keyData.router_settings,
      );
      if (routerSettings) {
        values.router_settings = routerSettings;
      }

      await onSubmit(withNormalizedEstimates(values));
    } finally {
      setIsKeySaving(false);
    }
  };

  const handleOrganizationChange = (setField: (value: string | undefined) => void, orgId: string | undefined) => {
    setField(orgId);
    setSelectedOrganizationId(orgId || null);
    form.setValue("team_id", undefined);
  };

  const handleTeamChange = (setField: (value: string | null) => void, teamId: string | null) => {
    setField(teamId);
    const selectedTeam = teams?.find((t) => t.team_id === teamId) || null;
    if (selectedTeam?.organization_id) {
      setSelectedOrganizationId(selectedTeam.organization_id);
      form.setValue("organization_id", selectedTeam.organization_id);
    } else if (!teamId) {
      setSelectedOrganizationId(null);
      form.setValue("organization_id", undefined);
    }
  };

  const handleDisabledCallbacksChange = (internalValues: string[]) => {
    setDisabledCallbacks(mapInternalToDisplayNames(internalValues));
    form.setValue("disabled_callbacks", internalValues);
  };

  const modelOptions = [
    ...modelSentinelOptions(keyData.team_id, team != null),
    ...availableModels.map((model) => ({
      value: model,
      label: model,
      disabled: hasAllModelsSentinel(selectedModels),
    })),
  ];

  const visibleTeams = selectedOrganizationId
    ? teams?.filter((t) => t.organization_id === selectedOrganizationId)
    : teams;

  return (
    <TooltipProvider>
      <form
        onSubmit={form.handleSubmit((values) =>
          handleSubmit(toSubmittedValues(values, { canViewPolicies, canViewPrompts })),
        )}
      >
        <FieldGroup>
          <FormField control={form.control} name="key_alias" label="Key Alias">
            {(field) => <Input {...field} value={(field.value as string | undefined) ?? ""} />}
          </FormField>

          <FormField
            control={form.control}
            name="models"
            label="Models"
            description={isModelsDisabled ? "Models field is disabled for this key type" : undefined}
          >
            {({ value, onChange, id }) => (
              <MultiSelect
                id={id}
                options={modelOptions}
                value={isModelsDisabled ? [] : (value as string[] | undefined) ?? []}
                onValueChange={(next) => {
                  if (next.includes("all-team-models")) {
                    onChange(["all-team-models"]);
                  } else if (next.includes("all-proxy-models")) {
                    onChange(["all-proxy-models"]);
                  } else {
                    onChange(next);
                  }
                }}
                disabled={isModelsDisabled}
                placeholder="Select models"
              />
            )}
          </FormField>

          <Field>
            <FieldLabel htmlFor={keyTypeFieldId}>Key Type</FieldLabel>
            <KeyTypeSelect
              id={keyTypeFieldId}
              value={keyTypeFromRoutes(allowedRoutes)}
              onChange={(value) => {
                switch (value) {
                  case "default":
                    form.setValue("allowed_routes", "");
                    break;
                  case "llm_api":
                    form.setValue("allowed_routes", "llm_api_routes");
                    break;
                  case "management":
                    form.setValue("allowed_routes", "management_routes");
                    form.setValue("models", []);
                    break;
                }
              }}
            />
          </Field>

          <FormField
            control={form.control}
            name="allowed_routes"
            label={labelWithHint(
              "Allowed Routes",
              "List of allowed routes for the key (comma-separated). Can be specific routes (e.g., '/chat/completions') or route patterns (e.g., 'llm_api_routes', 'management_routes', '/keys/*'). Leave empty to allow all routes.",
            )}
          >
            {(field) => (
              <Input
                {...field}
                value={(field.value as string | undefined) ?? ""}
                placeholder="Enter allowed routes (comma-separated). Special values: llm_api_routes, management_routes. Examples: llm_api_routes, /chat/completions, /keys/*. Leave empty to allow all routes"
              />
            )}
          </FormField>

          <FormField control={form.control} name="max_budget" label="Max Budget (USD)">
            {({ ref: _ref, ...field }) => (
              <NumericalInput
                {...field}
                value={field.value ?? ""}
                step={0.01}
                style={{ width: "100%" }}
                placeholder="Enter a numerical value"
              />
            )}
          </FormField>

          <FormField control={form.control} name="budget_duration" label="Reset Budget">
            {({ value, onChange, id }) => (
              <BudgetDurationDropdown
                id={id}
                value={value as string | null}
                onChange={(next) => onChange(next ?? null)}
                placeholder="Never resets"
              />
            )}
          </FormField>

          <Field>
            <FieldLabel>
              {labelWithHint(
                "Budget Windows",
                "Set multiple independent budget windows (e.g., hourly $10 AND monthly $200). Each window tracks spend separately and resets on its own schedule.",
              )}
            </FieldLabel>
            <BudgetWindowsEditor value={budgetLimits} onChange={setBudgetLimits} />
          </Field>

          <ModelMaxBudgetField
            key={keyData.token}
            premiumUser={premiumUser}
            value={modelBudget.value}
            onChange={modelBudget.setValue}
            availableModels={availableModels}
            usage={keyData.model_max_budget_usage}
            hint="Cap spend on individual models, each with its own reset window. Enforced across every request this key makes."
          />

          <Field>
            <FieldLabel>
              {labelWithHint(
                "Budget Fallbacks",
                "When a model exceeds its per-model budget, requests automatically reroute to fallback models instead of failing",
              )}
            </FieldLabel>
            <BudgetFallbacksEditor
              value={budgetFallbacks}
              onChange={setBudgetFallbacks}
              availableModels={availableModels}
            />
          </Field>

          <FormField control={form.control} name="tpm_limit" label="TPM Limit">
            {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
          </FormField>

          <FormField control={form.control} name="tpm_limit_type">
            {({ value, onChange, id }) => (
              <RateLimitTypeFormItem
                id={id}
                type="tpm"
                name="tpm_limit_type"
                showDetailedDescriptions={false}
                value={value as string | null}
                onChange={onChange}
              />
            )}
          </FormField>

          <FormField control={form.control} name="rpm_limit" label="RPM Limit">
            {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
          </FormField>

          <FormField control={form.control} name="rpm_limit_type">
            {({ value, onChange, id }) => (
              <RateLimitTypeFormItem
                id={id}
                type="rpm"
                name="rpm_limit_type"
                showDetailedDescriptions={false}
                value={value as string | null}
                onChange={onChange}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="throttle_on_budget_exceeded"
            label={labelWithHint(
              "Throttle on budget exceeded",
              "When this key exceeds its max budget, throttle its TPM/RPM to the globally configured percentage instead of blocking access entirely. Requires budget_exceeded_throttle_percentage in litellm_settings and a TPM/RPM limit on the key.",
            )}
          >
            {({ value, onChange, ref: _ref, ...field }) => (
              <Switch {...field} checked={Boolean(value)} onCheckedChange={onChange} />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="enable_prompt_caching"
            label={labelWithHint(
              "Enable Prompt Caching",
              "Automatically add prompt caching breakpoints (cache_control markers) to requests made with this key, cutting input cost on repeated prompts. Applies to Anthropic and Bedrock Claude models; requests that already set their own cache_control markers are left untouched.",
            )}
          >
            {({ value, onChange, ref: _ref, ...field }) => (
              <Switch {...field} checked={Boolean(value)} onCheckedChange={onChange} />
            )}
          </FormField>

          <FormField control={form.control} name="max_parallel_requests" label="Max Parallel Requests">
            {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
          </FormField>

          <FormField control={form.control} name="model_tpm_limit" label="Model TPM Limit">
            {(field) => (
              <Textarea
                {...field}
                value={(field.value as string | undefined) ?? ""}
                rows={4}
                placeholder='{"gpt-4": 100, "claude-v1": 200}'
              />
            )}
          </FormField>

          <FormField control={form.control} name="model_rpm_limit" label="Model RPM Limit">
            {(field) => (
              <Textarea
                {...field}
                value={(field.value as string | undefined) ?? ""}
                rows={4}
                placeholder='{"gpt-4": 100, "claude-v1": 200}'
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="default_estimated_output_tokens"
            label={labelWithHint("Estimated Output Tokens", estimateTooltip.estimate)}
          >
            {({ ref: _ref, ...field }) => (
              <NumericalInput {...field} value={field.value ?? ""} min={1} step={1} disabled={!canEditEstimates} />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="default_estimated_output_tokens_per_model"
            label={labelWithHint("Estimated Output Tokens Per Model", estimateTooltip.perModel)}
          >
            {(field) => (
              <Textarea
                {...field}
                value={(field.value as string | undefined) ?? ""}
                rows={4}
                placeholder='{"gpt-4": 4096}'
                disabled={!canEditEstimates}
              />
            )}
          </FormField>

          <Field>
            <FieldLabel>
              {labelWithHint(
                "Per-Tag Rate Limits",
                "Scope rate limits to a request tag so each tag (e.g. a cell or group) gets its own RPM counter. Requests without a matching tag fall back to the key-level limit.",
              )}
            </FieldLabel>
            <TagRateLimitEditor value={tagRateLimits} onChange={setTagRateLimits} />
          </Field>

          <FormField control={form.control} name="guardrails" label="Guardrails">
            {({ value, onChange }) =>
              accessToken ? (
                <GuardrailSelector
                  onChange={onChange}
                  value={value as string[] | undefined}
                  accessToken={accessToken}
                  disabled={!canEditGuardrails}
                />
              ) : (
                <div />
              )
            }
          </FormField>

          <FormField
            control={form.control}
            name="disable_global_guardrails"
            label={labelWithHint(
              "Disable Global Guardrails",
              "When enabled, this key will bypass any guardrails configured to run on every request (global guardrails)",
            )}
          >
            {({ value, onChange, ref: _ref, ...field }) => (
              <Switch {...field} checked={Boolean(value)} onCheckedChange={onChange} disabled={!canEditGuardrails} />
            )}
          </FormField>

          {canViewPolicies && (
            <FormField
              control={form.control}
              name="policies"
              label={labelWithHint("Policies", "Apply policies to this key to control guardrails and other settings")}
            >
              {({ value, onChange }) =>
                accessToken ? (
                  <PolicySelector
                    onChange={onChange}
                    value={value as string[] | undefined}
                    accessToken={accessToken}
                    disabled={!premiumUser}
                  />
                ) : (
                  <div />
                )
              }
            </FormField>
          )}

          <FormField control={form.control} name="tags" label="Tags">
            {({ value, onChange, id }) => (
              <TagsInput
                id={id}
                value={(value as string[] | undefined) ?? []}
                onValueChange={onChange}
                options={Object.values(tagsList).map((tag) => ({ value: tag.name, label: tag.name }))}
                placeholder="Select or enter tags"
              />
            )}
          </FormField>

          {canViewPrompts && (
            <FormField
              control={form.control}
              name="prompts"
              label={premiumUser ? "Prompts" : labelWithHint("Prompts", "Setting prompts by key is a premium feature")}
            >
              {({ value, onChange, id }) => (
                <TagsInput
                  id={id}
                  value={(value as string[] | undefined) ?? []}
                  onValueChange={onChange}
                  options={promptsList.map((name) => ({ value: name, label: name }))}
                  disabled={!premiumUser}
                  placeholder={currentValuePlaceholder(
                    premiumUser,
                    keyData.metadata?.prompts,
                    "Premium feature - Upgrade to set prompts by key",
                    "Select or enter prompts",
                  )}
                />
              )}
            </FormField>
          )}

          <FormField
            control={form.control}
            name="access_group_ids"
            label={labelWithHint(
              "Access Groups",
              "Assign access groups to this key. Access groups control which models, MCP servers, and agents this key can use",
            )}
          >
            {({ value, onChange }) => (
              <AccessGroupSelector
                value={value as string[] | undefined}
                onChange={onChange}
                placeholder="Select access groups (optional)"
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="allowed_passthrough_routes"
            label={
              premiumUser
                ? "Allowed Pass Through Routes"
                : labelWithHint(
                    "Allowed Pass Through Routes",
                    "Setting allowed pass through routes by key is a premium feature",
                  )
            }
          >
            {({ value, onChange }) => (
              <PassThroughRoutesSelector
                value={value as string[] | undefined}
                onChange={onChange}
                accessToken={accessToken || ""}
                placeholder={currentValuePlaceholder(
                  premiumUser,
                  keyData.metadata?.allowed_passthrough_routes,
                  "Premium feature - Upgrade to set allowed pass through routes by key",
                  "Select or enter allowed pass through routes",
                )}
                disabled={!premiumUser}
              />
            )}
          </FormField>

          <FormField control={form.control} name="vector_stores" label="Vector Stores">
            {({ value, onChange }) => (
              <VectorStoreSelector
                onChange={onChange}
                value={value as string[] | undefined}
                accessToken={accessToken || ""}
                placeholder="Select vector stores"
              />
            )}
          </FormField>

          <FormField control={form.control} name="mcp_servers_and_groups" label="MCP Servers / Access Groups">
            {({ value, onChange }) => (
              <MCPServerSelector
                onChange={onChange}
                value={value as McpServersAndGroups | undefined}
                accessToken={accessToken || ""}
                placeholder="Select MCP servers or access groups (optional)"
                allowNoMcpServers
              />
            )}
          </FormField>

          <div className="mb-6">
            <MCPToolPermissions
              accessToken={accessToken || ""}
              selectedServers={((mcpServersAndGroups as { servers?: string[] } | undefined)?.servers || []).filter(
                (s: string) => s !== NO_MCP_SERVERS_SENTINEL,
              )}
              toolPermissions={(mcpToolPermissions as Record<string, string[]> | undefined) || {}}
              onChange={(toolPerms) => form.setValue("mcp_tool_permissions", toolPerms)}
            />
          </div>

          <FormField control={form.control} name="agents_and_groups" label="Agents / Access Groups">
            {({ value, onChange }) => (
              <AgentSelector
                onChange={onChange}
                value={value as AgentsAndGroups | undefined}
                accessToken={accessToken || ""}
                placeholder="Select agents or access groups (optional)"
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="organization_id"
            label={labelWithHint(
              "Organization",
              "The organization this key belongs to. Selecting an organization filters the available teams.",
            )}
          >
            {({ value, onChange, id }) => (
              <OrganizationDropdown
                id={id}
                value={(value as string | undefined) ?? undefined}
                organizations={organizations}
                loading={isOrganizationsLoading}
                disabled={userRole !== "Admin"}
                onChange={(orgId) => handleOrganizationChange(onChange, orgId)}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="team_id"
            label="Team ID"
            description={
              enableProjectsUI && hasProject ? "Team is locked because this key belongs to a project" : undefined
            }
          >
            {({ value, onChange, id }) => (
              <Select
                value={(value as string | null) ?? null}
                onValueChange={(teamId: string | null) => handleTeamChange(onChange, teamId)}
                disabled={enableProjectsUI && hasProject}
                items={Object.fromEntries(
                  (visibleTeams ?? []).map((t) => [t.team_id, `${t.team_alias} (${t.team_id})`]),
                )}
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue placeholder="Select team" />
                </SelectTrigger>
                <SelectContent>
                  {visibleTeams?.map((t) => (
                    <SelectItem key={t.team_id} value={t.team_id}>
                      {`${t.team_alias} (${t.team_id})`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </FormField>

          {enableProjectsUI && hasProject && (
            <Field>
              <FieldLabel htmlFor={projectFieldId}>Project</FieldLabel>
              <Input id={projectFieldId} value={projectDisplay ?? ""} disabled readOnly />
            </Field>
          )}

          <Field>
            <FieldLabel>Router Settings</FieldLabel>
            <RouterSettingsAccordion
              ref={routerSettingsRef}
              accessToken={accessToken || ""}
              teamId={keyData.team_id}
              value={routerSettingsEditorValue(keyData.router_settings)}
            />
          </Field>

          <FormField control={form.control} name="logging_settings" label="Logging Settings">
            {({ value, onChange }) => (
              <EditLoggingSettings
                value={(value as unknown[] | undefined) ?? []}
                onChange={onChange}
                disabledCallbacks={disabledCallbacks}
                onDisabledCallbacksChange={handleDisabledCallbacksChange}
              />
            )}
          </FormField>

          <FormField control={form.control} name="metadata" label="Metadata">
            {(field) => <Textarea {...field} value={(field.value as string | undefined) ?? ""} rows={10} />}
          </FormField>

          <div className="mb-4">
            <FormField control={form.control} name="duration">
              {({ value, onChange, id }) => (
                <KeyLifecycleSettings
                  id={id}
                  value={(value as string | null) ?? ""}
                  onChange={onChange}
                  autoRotationEnabled={autoRotationEnabled}
                  onAutoRotationChange={setAutoRotationEnabled}
                  rotationInterval={rotationInterval}
                  onRotationIntervalChange={setRotationInterval}
                  neverExpire={neverExpire}
                  onNeverExpireChange={setNeverExpire}
                />
              )}
            </FormField>
          </div>
        </FieldGroup>

        <div className="sticky z-chrome bg-background p-4 border-t border-border -bottom-6 -inset-x-6">
          <div className="flex justify-end items-center gap-2">
            <Button type="button" variant="secondary" onClick={onCancel} disabled={isKeySaving}>
              Cancel
            </Button>
            <Button type="submit" disabled={isKeySaving} aria-busy={isKeySaving}>
              {isKeySaving && <UiLoadingSpinner className="size-4" />}
              Save Changes
            </Button>
          </div>
        </div>
      </form>
    </TooltipProvider>
  );
}
