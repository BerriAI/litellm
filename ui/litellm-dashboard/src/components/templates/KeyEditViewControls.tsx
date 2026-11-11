import { useTerminology } from "@/contexts/TerminologyContext";
import React from "react";
import { Control, UseFormReturn } from "react-hook-form";
import { useTranslation } from "react-i18next";
import type { ParseKeys, TFunction } from "i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CircleHelp } from "lucide-react";
import { FormField } from "@/components/shared/form/FormField";
import { toast } from "@/lib/toast";
import AgentSelector from "../agent_management/AgentSelector";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import { StorePromptsMetadataToggle } from "../common_components/StorePromptsToggle";
import NumericalInput from "../shared/numerical_input";
import SkillSelector from "../skills/SkillSelector";
import { moveTagsOutOfMetadataJson } from "./keyEditFieldNormalizers";
import { AgentsAndGroups, KeyEditFormValues } from "./keyEditFormValues";

export const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  </>
);

const KEY_TYPE_OPTIONS: { value: string; label: ParseKeys; hint: ParseKeys }[] = [
  {
    value: "default",
    label: "keyDetail.keyEditViewControls.keyType.fullAccess",
    hint: "keyDetail.keyEditViewControls.keyType.fullAccessHint",
  },
  {
    value: "llm_api",
    label: "keyDetail.keyEditViewControls.keyType.aiApis",
    hint: "keyDetail.keyEditViewControls.keyType.aiApisHint",
  },
  {
    value: "management",
    label: "keyDetail.keyEditViewControls.keyType.management",
    hint: "keyDetail.keyEditViewControls.keyType.managementHint",
  },
];

export const KeyTypeSelect = ({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
}) => {
  const { t } = useTranslation();
  const term = useTerminology();
  const keyTypeOptions = KEY_TYPE_OPTIONS.map((option) => ({
    ...option,
    label: term(t(option.label)),
    hint: term(t(option.hint)),
  }));
  return (
    <Select
      items={Object.fromEntries(keyTypeOptions.map((option) => [option.value, option.label]))}
      value={value}
      onValueChange={(next: string | null) => next != null && onChange(next)}
    >
      <SelectTrigger id={id} className="w-full">
        <SelectValue placeholder={t("keyDetail.keyEditViewControls.keyType.placeholder")} />
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
  );
};

const SKILLS_HINT: ParseKeys = "keyDetail.keyEditViewControls.hints.skills";

const TPD_HINT: ParseKeys = "keyDetail.keyEditViewControls.hints.tpd";

export const KeyRateLimitFields = ({ control }: { control: Control<KeyEditFormValues> }) => {
  const { t } = useTranslation();
  return (
    <>
      <FormField control={control} name="tpm_limit" label={t("keyDetail.keyEditViewControls.rateLimits.tpmLimit")}>
        {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
      </FormField>

      <FormField control={control} name="tpm_limit_type">
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

      <FormField control={control} name="rpm_limit" label={t("keyDetail.keyEditViewControls.rateLimits.rpmLimit")}>
        {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
      </FormField>

      <FormField control={control} name="rpm_limit_type">
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
        control={control}
        name="tpd_limit"
        label={labelWithHint(t("keyDetail.keyEditViewControls.rateLimits.tpdLimit"), t(TPD_HINT))}
      >
        {({ ref: _ref, ...field }) => <NumericalInput {...field} value={field.value ?? ""} min={0} />}
      </FormField>
    </>
  );
};

export const KeyAgentAndSkillFields = ({
  control,
  accessToken,
}: {
  control: Control<KeyEditFormValues>;
  accessToken: string;
}) => {
  const { t } = useTranslation();
  return (
    <>
      <FormField control={control} name="agents_and_groups" label={t("keyDetail.keyEditViewControls.agents.label")}>
        {({ value, onChange }) => (
          <AgentSelector
            onChange={onChange}
            value={value as AgentsAndGroups | undefined}
            accessToken={accessToken}
            placeholder={t("keyDetail.keyEditViewControls.agents.placeholder")}
          />
        )}
      </FormField>

      <FormField
        control={control}
        name="skills"
        label={labelWithHint(t("keyDetail.keyEditViewControls.skills.label"), t(SKILLS_HINT))}
      >
        {({ value, onChange }) => (
          <SkillSelector onChange={onChange} value={value as string[] | undefined} accessToken={accessToken} />
        )}
      </FormField>
    </>
  );
};

type KeyEditForm = Pick<
  UseFormReturn<KeyEditFormValues, unknown, KeyEditFormValues>,
  "control" | "getValues" | "setValue"
>;

export const moveMetadataTagsToTagsField = (form: KeyEditForm, t: TFunction): void => {
  const moved = moveTagsOutOfMetadataJson(form.getValues("metadata"), form.getValues("tags"));
  if (moved === null) return;
  form.setValue("metadata", moved.metadata, { shouldDirty: true });
  form.setValue("tags", moved.tags, { shouldDirty: true });
  if (moved.movedTags.length > 0) {
    toast.info(t("keyDetail.keyEditViewControls.metadata.movedTags", { tags: moved.movedTags.join(", ") }));
  }
};

export const KeyMetadataField = ({ form }: { form: KeyEditForm }) => {
  const { t } = useTranslation();
  return (
    <FormField
      control={form.control}
      name="metadata"
      label={t("keyDetail.keyEditViewControls.metadata.label")}
      description={t("keyDetail.keyEditViewControls.metadata.description")}
    >
      {(field) => (
        <>
          <StorePromptsMetadataToggle value={field.value as string | undefined} onChange={field.onChange} />
          <Textarea
            {...field}
            value={(field.value as string | undefined) ?? ""}
            rows={10}
            onBlur={() => {
              field.onBlur();
              moveMetadataTagsToTagsField(form, t);
            }}
          />
        </>
      )}
    </FormField>
  );
};

export const KeyBudgetNumberField = ({
  control,
  name,
  label,
  placeholder,
}: {
  control: Control<KeyEditFormValues>;
  name: "max_budget" | "soft_budget";
  label: string;
  placeholder: string;
}) => (
  <FormField control={control} name={name} label={label}>
    {({ ref: _ref, ...field }) => (
      <NumericalInput
        {...field}
        value={field.value ?? ""}
        step={0.01}
        style={{ width: "100%" }}
        placeholder={placeholder}
      />
    )}
  </FormField>
);
