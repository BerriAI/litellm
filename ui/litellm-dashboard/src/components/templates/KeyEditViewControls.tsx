import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Control, UseFormReturn } from "react-hook-form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CircleHelp } from "lucide-react";
import { FormField } from "@/components/shared/form/FormField";
import { toast } from "@/lib/toast";
import TranslatedMessage from "../shared/TranslatedMessage";
import AgentSelector from "../agent_management/AgentSelector";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import NumericalInput from "../shared/numerical_input";
import SkillSelector from "../skills/SkillSelector";
import { moveTagsOutOfMetadataJson } from "./keyEditFieldNormalizers";
import { AgentsAndGroups, KeyEditFormValues } from "./keyEditFormValues";
import { estimateChecks, estimateTooltips } from "./estimatedOutputTokens";
import type { Translate } from "@/i18n/language";

export const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  </>
);

const KEY_TYPE_OPTIONS = [
  { value: "default", label: "Full Access", hint: "Can call all routes (AI APIs, Management, and read-only)" },
  { value: "llm_api", label: "AI APIs", hint: "Can call only AI API routes (chat/completions, embeddings, etc.)" },
  { value: "management", label: "Management", hint: "Can call only management routes (user/team/key management)" },
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
  const { t } = useLanguage();
  return (
    <Select
      items={Object.fromEntries(KEY_TYPE_OPTIONS.map((option) => [option.value, t(option.label)]))}
      value={value}
      onValueChange={(next: string | null) => next != null && onChange(next)}
    >
      <SelectTrigger id={id} className="w-full">
        <SelectValue placeholder={t("Select key type")} />
      </SelectTrigger>
      <SelectContent>
        {KEY_TYPE_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            <div className="py-1">
              <div className="font-medium">{t(option.label)}</div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">{t(option.hint)}</div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

const SKILLS_HINT =
  "Enabled skills are visible to every key. Grant disabled (private) Claude Code plugins to this key here.";

const TPD_HINT =
  "Daily token budget for batch submissions (/v1/batches). When set, batch input files are charged against this 24h window instead of the key's TPM/RPM limits. Online requests keep using TPM/RPM.";

export const KeyRateLimitFields = ({ control }: { control: Control<KeyEditFormValues> }) => {
  const { t } = useLanguage();
  return (
    <>
      <FormField control={control} name="tpm_limit" label={t("TPM Limit")}>
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

      <FormField control={control} name="rpm_limit" label={t("RPM Limit")}>
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

      <FormField control={control} name="tpd_limit" label={labelWithHint(t("TPD Limit (batch)"), t(TPD_HINT))}>
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
  const { t } = useLanguage();
  return (
    <>
      <FormField control={control} name="agents_and_groups" label={t("Agents / Access Groups")}>
        {({ value, onChange }) => (
          <AgentSelector
            onChange={onChange}
            value={value as AgentsAndGroups | undefined}
            accessToken={accessToken}
            placeholder={t("Select agents or access groups (optional)")}
          />
        )}
      </FormField>

      <FormField control={control} name="skills" label={labelWithHint(t("Skills"), t(SKILLS_HINT))}>
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

// 保留旧调用的翻译参数兼容性，通知在实际显示时读取语言。
export const moveMetadataTagsToTagsField = (form: KeyEditForm, _t?: Translate): void => {
  const moved = moveTagsOutOfMetadataJson(form.getValues("metadata"), form.getValues("tags"));
  if (moved === null) return;
  form.setValue("metadata", moved.metadata, { shouldDirty: true });
  form.setValue("tags", moved.tags, { shouldDirty: true });
  if (moved.movedTags.length > 0) {
    toast.info(
      <TranslatedMessage
        message="Moved {tags} from metadata to the Tags field"
        values={{ tags: moved.movedTags.join(", ") }}
      />,
    );
  }
};

export const KeyMetadataField = ({ form }: { form: KeyEditForm }) => {
  const { t } = useLanguage();
  return (
    <FormField
      control={form.control}
      name="metadata"
      label={t("Metadata")}
      description={t("Tags are managed by the Tags field above. A tags array typed here is moved to that field.")}
    >
      {(field) => (
        <Textarea
          {...field}
          value={(field.value as string | undefined) ?? ""}
          rows={10}
          onBlur={() => {
            field.onBlur();
            moveMetadataTagsToTagsField(form, t);
          }}
        />
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
}) => {
  return (
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
};

// 预计输出 Token 是一个独立字段组，仅展示文案随语言变化，不改变 TPM 配置值。
export const KeyEstimatedOutputFields = ({
  control,
  canEdit,
}: {
  control: Control<KeyEditFormValues>;
  canEdit: boolean;
}) => {
  const { t } = useLanguage();
  const tooltips = estimateTooltips(canEdit);
  return (
    <>
      <FormField
        control={control}
        name="default_estimated_output_tokens"
        label={labelWithHint(t("Estimated Output Tokens"), t(tooltips.estimate))}
        renderError={(error) =>
          error.message === estimateChecks.positive.message ? t(estimateChecks.positive.message) : error.message
        }
      >
        {({ ref: _ref, ...field }) => (
          <NumericalInput {...field} value={field.value ?? ""} min={1} step={1} disabled={!canEdit} />
        )}
      </FormField>
      <FormField
        control={control}
        name="default_estimated_output_tokens_per_model"
        label={labelWithHint(t("Estimated Output Tokens Per Model"), t(tooltips.perModel))}
        renderError={(error) =>
          error.message === estimateChecks.perModel.message ? t(estimateChecks.perModel.message) : error.message
        }
      >
        {(field) => (
          <Textarea
            {...field}
            value={(field.value as string | undefined) ?? ""}
            rows={4}
            placeholder='{"gpt-4": 4096}'
            disabled={!canEdit}
          />
        )}
      </FormField>
    </>
  );
};

// 按模型限速的 JSON 字段保持原有协议，只将标签随界面语言切换。
export const KeyPerModelRateLimitFields = ({ control }: { control: Control<KeyEditFormValues> }) => {
  const { t } = useLanguage();
  return (
    <>
      <FormField control={control} name="model_tpm_limit" label={t("Model TPM Limit")}>
        {(field) => (
          <Textarea
            {...field}
            value={(field.value as string | undefined) ?? ""}
            rows={4}
            placeholder='{"gpt-4": 100, "claude-v1": 200}'
          />
        )}
      </FormField>
      <FormField control={control} name="model_rpm_limit" label={t("Model RPM Limit")}>
        {(field) => (
          <Textarea
            {...field}
            value={(field.value as string | undefined) ?? ""}
            rows={4}
            placeholder='{"gpt-4": 100, "claude-v1": 200}'
          />
        )}
      </FormField>
    </>
  );
};
