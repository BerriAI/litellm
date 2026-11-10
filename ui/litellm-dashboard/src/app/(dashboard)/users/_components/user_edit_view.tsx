import type { TFunction } from "i18next";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import { all_admin_roles } from "@/utils/roles";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { ModelMaxBudget, ModelMaxBudgetField } from "@/components/key_team_helpers/ModelMaxBudgetEditor";
import { modelMaxBudgetUpdate } from "@/components/key_team_helpers/modelMaxBudgetPayload";
import { useSeededState } from "@/components/key_team_helpers/useSeededState";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import type { ObjectPermission } from "@/components/object_permission_types";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import { CircleHelp } from "lucide-react";

const isBlank = (value: string | number | null | undefined): boolean =>
  value === null || value === undefined || String(value).trim() === "";
const rateLimitField = (t: TFunction) => {
  const rateLimitError = t("users.editView.errors.rateLimit");
  return z
    .union([z.string(), z.number()])
    .nullish()
    .transform((value) => (isBlank(value) ? null : Number(value)))
    .pipe(z.number({ error: rateLimitError }).int(rateLimitError).nonnegative(rateLimitError).nullable());
};

interface UserEditViewProps {
  userData: any;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  teams: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  userModels: string[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  isBulkEdit?: boolean;
  objectPermission?: ObjectPermission | null;
  premiumUser?: boolean;
}

const MCP_SELECTION_SHAPE = z.object({
  servers: z.array(z.string()),
  accessGroups: z.array(z.string()),
  toolsets: z.array(z.string()),
});

// The proxy stores unset user fields as null rather than leaving them out, and
// antd forwarded whatever it was handed, so nullish is what keeps a loaded user
// editable at all.
const userEditShape = (t: TFunction) => ({
  user_id: z.string().nullish(),
  user_email: z.string().nullish(),
  user_alias: z.string().nullish(),
  user_role: z.string().nullish(),
  models: z.array(z.string()),
  budget_duration: z.string().nullish(),
  metadata: z.string().nullish(),
  tpm_limit: rateLimitField(t),
  rpm_limit: rateLimitField(t),
  mcp_servers_and_groups: MCP_SELECTION_SHAPE.optional(),
  mcp_tool_permissions: z.record(z.string(), z.array(z.string())).optional(),
});

const userEditSchema = (unlimitedBudget: boolean, t: TFunction) =>
  z.object({
    ...userEditShape(t),
    max_budget: z
      .union([z.string(), z.number()])
      .nullish()
      .refine((value) => unlimitedBudget || !isBlank(value), t("users.editView.errors.budgetRequired")),
  });

type UserEditFormInput = z.input<ReturnType<typeof userEditSchema>>;
type UserEditFormValues = z.output<ReturnType<typeof userEditSchema>>;

const buildMcpFieldValues = (objectPermission: ObjectPermission | null | undefined) => ({
  mcp_servers_and_groups: {
    servers: objectPermission?.mcp_servers ?? [],
    accessGroups: objectPermission?.mcp_access_groups ?? [],
    toolsets: objectPermission?.mcp_toolsets ?? [],
  },
  mcp_tool_permissions: objectPermission?.mcp_tool_permissions ?? {},
});

// antd only reported the fields that were actually mounted, so the identity and
// MCP keys have to stay out of the store entirely when their controls are not
// rendered, or the request grows keys the previous form never sent.
const toFormValues = (
  userData: any,
  objectPermission: ObjectPermission | null | undefined,
  isBulkEdit: boolean,
  canEditMcpPermissions: boolean,
): UserEditFormInput => {
  const maxBudget = userData.user_info?.max_budget;
  const isUnlimited = maxBudget === null || maxBudget === undefined;
  return {
    ...(isBulkEdit
      ? {}
      : {
          user_id: userData.user_id,
          user_email: userData.user_info?.user_email,
          tpm_limit: userData.user_info?.tpm_limit ?? "",
          rpm_limit: userData.user_info?.rpm_limit ?? "",
        }),
    user_alias: userData.user_info?.user_alias,
    user_role: userData.user_info?.user_role,
    models: userData.user_info?.models || [],
    max_budget: isUnlimited ? "" : maxBudget,
    budget_duration: userData.user_info?.budget_duration,
    metadata: userData.user_info?.metadata ? JSON.stringify(userData.user_info.metadata, null, 2) : undefined,
    ...(canEditMcpPermissions ? buildMcpFieldValues(objectPermission) : {}),
  };
};

type ParsedMetadata = { ok: true; value: unknown } | { ok: false };

const parseMetadata = (metadata: string | null | undefined): ParsedMetadata => {
  if (!metadata) {
    return { ok: true, value: metadata };
  }
  try {
    return { ok: true, value: JSON.parse(metadata) };
  } catch (error) {
    console.error("Error parsing metadata JSON:", error);
    return { ok: false };
  }
};

const changedLimit = (value: number | null, stored: number | null | undefined): number | null | undefined =>
  value === (stored ?? null) ? undefined : value;

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

export function UserEditView({
  userData,
  onCancel,
  onSubmit,
  teams,
  accessToken,
  userID,
  userRole,
  userModels,
  possibleUIRoles,
  isBulkEdit = false,
  objectPermission,
  premiumUser = false,
}: UserEditViewProps) {
  const { t } = useTranslation();
  const canEditMcpPermissions = !isBulkEdit && all_admin_roles.includes(userRole || "");
  const [unlimitedBudget, setUnlimitedBudget] = useState(false);
  const [modelMaxBudget, setModelMaxBudget] = useSeededState<ModelMaxBudget>(
    userData.user_id,
    () => userData.user_info?.model_max_budget ?? {},
  );
  const schema = useMemo(() => userEditSchema(unlimitedBudget, t), [unlimitedBudget, t]);
  const form = useZodForm(schema, {
    defaultValues: toFormValues(userData, objectPermission, isBulkEdit, canEditMcpPermissions),
  });

  React.useEffect(() => {
    const maxBudget = userData.user_info?.max_budget;
    setUnlimitedBudget(maxBudget === null || maxBudget === undefined);
    form.reset(toFormValues(userData, objectPermission, isBulkEdit, canEditMcpPermissions));
  }, [userData, objectPermission, canEditMcpPermissions, isBulkEdit, form]);

  const handleUnlimitedBudgetChange = (checked: boolean) => {
    setUnlimitedBudget(checked);
    if (checked) {
      form.setValue("max_budget", "");
    }
  };

  const handleSubmit = (values: UserEditFormValues) => {
    const metadata = parseMetadata(values.metadata);
    if (!metadata.ok) {
      return;
    }

    const { tpm_limit: tpmLimitInput, rpm_limit: rpmLimitInput, ...formValues } = values;
    const modelBudgets = modelMaxBudgetUpdate(modelMaxBudget, userData.user_info?.model_max_budget);
    const tpmLimit = changedLimit(tpmLimitInput, userData.user_info?.tpm_limit);
    const rpmLimit = changedLimit(rpmLimitInput, userData.user_info?.rpm_limit);
    const payload = {
      ...formValues,
      ...("metadata" in values ? { metadata: metadata.value } : {}),
      ...(modelBudgets !== undefined && { model_max_budget: modelBudgets }),
      ...(tpmLimit !== undefined && { tpm_limit: tpmLimit }),
      ...(rpmLimit !== undefined && { rpm_limit: rpmLimit }),
      max_budget:
        unlimitedBudget || values.max_budget === "" || values.max_budget === undefined ? null : values.max_budget,
    };
    onSubmit(payload);
  };

  const modelOptions = [
    { label: t("users.editView.modelOptions.allProxyModels"), value: "all-proxy-models" },
    { label: t("users.editView.modelOptions.noDefaultModels"), value: "no-default-models" },
    ...userModels.map((model) => ({ label: getModelDisplayName(model), value: model })),
  ];

  const roleOptions = Object.entries(possibleUIRoles ?? {}).map(([role, { ui_label, description }]) => ({
    value: role,
    label: ui_label,
    description,
  }));

  return (
    <TooltipProvider>
      <form onSubmit={form.handleSubmit(handleSubmit)}>
        <FieldGroup>
          {!isBulkEdit && (
            <FormField control={form.control} name="user_id" label={t("users.editView.fields.userId")}>
              {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} disabled />}
            </FormField>
          )}

          {!isBulkEdit && (
            <FormField control={form.control} name="user_email" label={t("users.editView.fields.email")}>
              {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} />}
            </FormField>
          )}

          <FormField control={form.control} name="user_alias" label={t("users.editView.fields.userAlias")}>
            {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} />}
          </FormField>

          <FormField
            control={form.control}
            name="user_role"
            label={labelWithHint(
              t("users.editView.fields.globalProxyRole"),
              t("users.editView.fields.globalProxyRoleHint"),
            )}
          >
            {({ id, value, onChange }) => (
              <Select
                items={roleOptions}
                value={value === undefined || value === "" ? null : value}
                onValueChange={(selected: string | null) => onChange(selected ?? undefined)}
              >
                <SelectTrigger id={id} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {roleOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <span>{option.label}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{option.description}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </FormField>

          <FormField
            control={form.control}
            name="models"
            label={labelWithHint(
              t("users.editView.fields.personalModels"),
              t("users.editView.fields.personalModelsHint"),
            )}
          >
            {({ value, onChange }) => (
              <MultiSelect
                options={modelOptions}
                value={value}
                onValueChange={onChange}
                placeholder={t("users.editView.fields.modelsPlaceholder")}
                disabled={!all_admin_roles.includes(userRole || "")}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="max_budget"
            label={
              <>
                {t("users.editView.fields.maxBudget")}
                <label className="ml-3 inline-flex items-center gap-2 font-normal">
                  <Checkbox checked={unlimitedBudget} onCheckedChange={handleUnlimitedBudgetChange} />
                  {t("users.editView.fields.unlimitedBudget")}
                </label>
              </>
            }
          >
            {({ ref, value, onChange, ...control }) => (
              <Input
                {...control}
                ref={ref}
                type="number"
                step={0.01}
                value={value ?? ""}
                onChange={(event) => onChange(event.target.value)}
                onWheel={(event) => event.currentTarget.blur()}
                placeholder={t("users.editView.fields.maxBudgetPlaceholder")}
                disabled={unlimitedBudget}
              />
            )}
          </FormField>

          <FormField control={form.control} name="budget_duration" label={t("users.editView.fields.resetBudget")}>
            {({ id, value, onChange }) => (
              <BudgetDurationDropdown
                id={id}
                value={value}
                onChange={onChange}
                placeholder={isBulkEdit ? t("users.editView.fields.noChange") : undefined}
              />
            )}
          </FormField>

          {!isBulkEdit && (
            <>
              <FormField
                control={form.control}
                name="tpm_limit"
                label={labelWithHint(t("users.editView.fields.tpmLimit"), t("users.editView.fields.rateLimitHint"))}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    step={1}
                    value={value ?? ""}
                    onChange={(event) => onChange(event.target.value)}
                    onWheel={(event) => event.currentTarget.blur()}
                    placeholder={t("users.editView.fields.rateLimitPlaceholder")}
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="rpm_limit"
                label={labelWithHint(t("users.editView.fields.rpmLimit"), t("users.editView.fields.rateLimitHint"))}
              >
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    step={1}
                    value={value ?? ""}
                    onChange={(event) => onChange(event.target.value)}
                    onWheel={(event) => event.currentTarget.blur()}
                    placeholder={t("users.editView.fields.rateLimitPlaceholder")}
                  />
                )}
              </FormField>
            </>
          )}

          {/* Bulk edit forwards a fixed field list and has no single stored budget to
              diff against, so the editor would silently discard whatever was typed. */}
          {!isBulkEdit && (
            <ModelMaxBudgetField
              key={userData.user_id}
              premiumUser={premiumUser}
              value={modelMaxBudget}
              onChange={setModelMaxBudget}
              availableModels={userModels}
              usage={userData.user_info?.model_max_budget_usage}
              hint={t("users.editView.fields.modelMaxBudgetHint")}
            />
          )}

          <FormField control={form.control} name="metadata" label={t("users.editView.fields.metadata")}>
            {({ ref, value, ...control }) => (
              <Textarea
                {...control}
                ref={ref}
                value={value ?? ""}
                rows={4}
                placeholder={t("users.editView.fields.metadataPlaceholder")}
              />
            )}
          </FormField>

          {canEditMcpPermissions && (
            <>
              <FormField
                control={form.control}
                name="mcp_servers_and_groups"
                label={labelWithHint(t("users.editView.fields.mcpServers"), t("users.editView.fields.mcpServersHint"))}
              >
                {({ value, onChange }) => (
                  <MCPServerSelector
                    onChange={onChange}
                    value={value}
                    accessToken={accessToken || ""}
                    userId={isBulkEdit ? null : userData.user_id}
                    placeholder={t("users.editView.fields.mcpServersPlaceholder")}
                  />
                )}
              </FormField>

              <MCPToolPermissions
                accessToken={accessToken || ""}
                selectedServers={form.watch("mcp_servers_and_groups")?.servers || []}
                selectedAccessGroups={form.watch("mcp_servers_and_groups")?.accessGroups || []}
                selectedToolsets={form.watch("mcp_servers_and_groups")?.toolsets || []}
                toolPermissions={form.watch("mcp_tool_permissions") || {}}
                onChange={(toolPerms) => form.setValue("mcp_tool_permissions", toolPerms)}
              />
            </>
          )}
        </FieldGroup>

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="secondary" type="button" onClick={onCancel}>
            {t("users.editView.actions.cancel")}
          </Button>
          <Button type="submit">{t("users.editView.actions.save")}</Button>
        </div>
      </form>
    </TooltipProvider>
  );
}
