import React from "react";
import { Input, Select, Tooltip, Typography } from "antd";
import { Button } from "@/components/ui/button";
import { CircleMinus, Info, Plus } from "lucide-react";
import { useFieldArray, useFormContext, useWatch } from "react-hook-form";

import {
  MountedFormField,
  useMountedName,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { antdRequired } from "@/components/common_components/antdFormRules";
import { matchesPattern, selectControl, textControl } from "./mcpFieldRules";
import { listControl } from "./mcpFormStore";

const { Text } = Typography;

const SCOPE_OPTIONS = [
  { value: "global", label: "Instance" },
  { value: "user", label: "Per-user" },
];

const VARIABLE_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

/**
 * Form section for admin-configured MCP environment variables.
 *
 * Each row has: name | value | scope. Variables can be interpolated into
 * Static Headers via ${NAME}. ``scope=global`` (shown as "Instance") values
 * are used as-is. ``scope=user`` (shown as "Per-user") values are filled in
 * by each user via the MCP Gateway dashboard.
 *
 * The parent form reads the ``env_vars`` field from the form values.
 */
const EnvVarsSection: React.FC = () => {
  const { control } = useFormContext<MountedFormValues>();
  const { fields, append, remove } = useFieldArray({ control: listControl(control), name: "env_vars" });
  useMountedName("env_vars");

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
      <div className="flex items-center gap-2 mb-1">
        <Text strong className="text-sm">
          Variables
        </Text>
        <Tooltip
          title={
            <>
              Define variables you can interpolate in Static Headers or Authentication using{" "}
              <code>{"${VAR_NAME}"}</code>. <br />
              <b>Instance</b>: admin-defined value used for every user.
              <br />
              <b>Per-user</b>: each user supplies their own value (e.g. personal credentials) via the MCP Gateway
              dashboard.
            </>
          }
        >
          <Info className="size-4 text-blue-400 hover:text-blue-600 cursor-help" />
        </Tooltip>
      </div>
      <Text className="text-xs text-gray-600 block mb-3">
        Reference these in Static Headers or Authentication as <code>{"${VAR_NAME}"}</code>. For example:{" "}
        <code className="bg-white px-1 rounded-sm border border-gray-200">
          {"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}"}
        </code>
      </Text>

      <div className="space-y-2">
        {fields.length > 0 && (
          <div className="flex gap-3 px-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
            <div style={{ flex: 1 }}>Variable Name</div>
            <div style={{ flex: 1 }}>Value / Description</div>
            <div style={{ width: 160 }}>Scope</div>
            <div style={{ width: 24 }} />
          </div>
        )}
        {fields.map((item, index) => (
          <div key={item.id} className="flex gap-3 items-start">
            <MountedFormField
              name={["env_vars", String(index), "name"]}
              className="mb-0 flex-1"
              rules={{
                validate: {
                  required: antdRequired("Variable name is required"),
                  pattern: matchesPattern(
                    VARIABLE_NAME_PATTERN,
                    "Use letters, digits, underscores; cannot start with a digit.",
                  ),
                },
              }}
            >
              {(control) => (
                <Input {...textControl(control)} placeholder="e.g. DB_PROTOCOL" className="rounded-md font-mono" />
              )}
            </MountedFormField>
            <div style={{ flex: 1 }}>
              <ScopedValueOrDescription index={index} />
            </div>
            <MountedFormField name={["env_vars", String(index), "scope"]} className="mb-0 w-40" defaultValue="global">
              {(control) => <Select {...selectControl<string>(control)} options={SCOPE_OPTIONS} />}
            </MountedFormField>
            <div style={{ width: 24, height: 32 }} className="flex items-center justify-center">
              <CircleMinus
                onClick={() => remove(index)}
                className="size-4 text-gray-500 hover:text-red-500 cursor-pointer"
              />
            </div>
          </div>
        ))}
        <Button variant="outline" className="w-full border-dashed" onClick={() => append({ scope: "global" })}>
          <Plus />
          Add Variable
        </Button>
      </div>
    </div>
  );
};

// For instance-scoped vars this column holds the admin value. For per-user
// vars the value comes from each user later, so the column instead captures an
// optional description that the per-user fill-in modal shows as a hint.
const ScopedValueOrDescription: React.FC<{ index: number }> = ({ index }) => {
  const isPerUser = useWatch({ name: `env_vars.${index}.scope` }) === "user";
  if (isPerUser) {
    return (
      <MountedFormField name={["env_vars", String(index), "description"]} className="mb-0">
        {(control) => (
          <Input
            {...textControl(control)}
            addonBefore={
              <Tooltip title="Per-user variables have no shared value. This text is only a hint shown to each user when they fill in their own value.">
                <span className="text-xs text-gray-500 cursor-help whitespace-nowrap">
                  <Info className="mr-1 inline size-3 align-text-bottom" />
                  Hint
                </span>
              </Tooltip>
            }
            placeholder="e.g. Your DB username"
            styles={{ input: { color: "#9ca3af" } }}
          />
        )}
      </MountedFormField>
    );
  }
  return (
    <MountedFormField name={["env_vars", String(index), "value"]} className="mb-0">
      {(control) => <Input {...textControl(control)} placeholder="e.g. postgresql" className="rounded-md font-mono" />}
    </MountedFormField>
  );
};

export default EnvVarsSection;
