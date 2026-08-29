import React, { useMemo, useState } from "react";
import { z } from "zod/v4";
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
const userEditShape = {
  user_id: z.string().nullish(),
  user_email: z.string().nullish(),
  user_alias: z.string().nullish(),
  user_role: z.string().nullish(),
  models: z.array(z.string()),
  budget_duration: z.string().nullish(),
  metadata: z.string().nullish(),
  mcp_servers_and_groups: MCP_SELECTION_SHAPE.optional(),
  mcp_tool_permissions: z.record(z.string(), z.array(z.string())).optional(),
};

const budgetSchema = (unlimitedBudget: boolean) =>
  z.object({
    ...userEditShape,
    max_budget: z
      .union([z.string(), z.number()])
      .nullish()
      .refine(
        (value) => unlimitedBudget || (value !== "" && value !== null && value !== undefined),
        "Please enter a budget or select Unlimited Budget",
      ),
  });

type UserEditFormValues = z.infer<ReturnType<typeof budgetSchema>>;

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
): UserEditFormValues => {
  const maxBudget = userData.user_info?.max_budget;
  const isUnlimited = maxBudget === null || maxBudget === undefined;
  return {
    ...(isBulkEdit ? {} : { user_id: userData.user_id, user_email: userData.user_info?.user_email }),
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
  const canEditMcpPermissions = !isBulkEdit && all_admin_roles.includes(userRole || "");
  const [unlimitedBudget, setUnlimitedBudget] = useState(false);
  const [modelMaxBudget, setModelMaxBudget] = useSeededState<ModelMaxBudget>(
    userData.user_id,
    () => userData.user_info?.model_max_budget ?? {},
  );
  const schema = useMemo(() => budgetSchema(unlimitedBudget), [unlimitedBudget]);
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

    const modelBudgets = modelMaxBudgetUpdate(modelMaxBudget, userData.user_info?.model_max_budget);
    onSubmit({
      ...values,
      ...("metadata" in values ? { metadata: metadata.value } : {}),
      ...(modelBudgets !== undefined && { model_max_budget: modelBudgets }),
      max_budget:
        unlimitedBudget || values.max_budget === "" || values.max_budget === undefined ? null : values.max_budget,
    });
  };

  const modelOptions = [
    { label: "All Proxy Models", value: "all-proxy-models" },
    { label: "No Default Models", value: "no-default-models" },
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
            <FormField control={form.control} name="user_id" label="User ID">
              {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} disabled />}
            </FormField>
          )}

          {!isBulkEdit && (
            <FormField control={form.control} name="user_email" label="Email">
              {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} />}
            </FormField>
          )}

          <FormField control={form.control} name="user_alias" label="User Alias">
            {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} />}
          </FormField>

          <FormField
            control={form.control}
            name="user_role"
            label={labelWithHint(
              "Global Proxy Role",
              "This is the role that the user will globally on the proxy. This role is independent of any team/org specific roles.",
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
              "Personal Models",
              "Select which models this user can access outside of team-scope. Choose 'All Proxy Models' to grant access to all models available on the proxy.",
            )}
          >
            {({ value, onChange }) => (
              <MultiSelect
                options={modelOptions}
                value={value}
                onValueChange={onChange}
                placeholder="Select models"
                disabled={!all_admin_roles.includes(userRole || "")}
              />
            )}
          </FormField>

          <FormField
            control={form.control}
            name="max_budget"
            label={
              <>
                Max Budget (USD)
                <label className="ml-3 inline-flex items-center gap-2 font-normal">
                  <Checkbox checked={unlimitedBudget} onCheckedChange={handleUnlimitedBudgetChange} />
                  Unlimited Budget
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
                placeholder="Enter a numerical value"
                disabled={unlimitedBudget}
              />
            )}
          </FormField>

          <FormField control={form.control} name="budget_duration" label="Reset Budget">
            {({ id, value, onChange }) => <BudgetDurationDropdown id={id} value={value} onChange={onChange} />}
          </FormField>

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
              hint="Cap this user's spend on individual models, each with its own reset window. Applies across every key the user holds."
            />
          )}

          <FormField control={form.control} name="metadata" label="Metadata">
            {({ ref, value, ...control }) => (
              <Textarea {...control} ref={ref} value={value ?? ""} rows={4} placeholder="Enter metadata as JSON" />
            )}
          </FormField>

          {canEditMcpPermissions && (
            <>
              <FormField
                control={form.control}
                name="mcp_servers_and_groups"
                label={labelWithHint(
                  "MCP Servers / Access Groups",
                  "Caps which MCP servers, access groups, and tools this user may reach. Every key the user holds is limited to this set.",
                )}
              >
                {({ value, onChange }) => (
                  <MCPServerSelector
                    onChange={onChange}
                    value={value}
                    accessToken={accessToken || ""}
                    placeholder="Select MCP servers or access groups (optional)"
                  />
                )}
              </FormField>

              <MCPToolPermissions
                accessToken={accessToken || ""}
                selectedServers={form.watch("mcp_servers_and_groups")?.servers || []}
                toolPermissions={form.watch("mcp_tool_permissions") || {}}
                onChange={(toolPerms) => form.setValue("mcp_tool_permissions", toolPerms)}
              />
            </>
          )}
        </FieldGroup>

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="secondary" type="button" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">Save Changes</Button>
        </div>
      </form>
    </TooltipProvider>
  );
}
