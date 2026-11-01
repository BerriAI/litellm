import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { ParseKeys } from "i18next";
import { Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

export interface RequestTagMapping {
  header: "baggage";
  key: string;
  value?: string | null;
  tag?: string | null;
  tag_prefix?: string | null;
  required?: boolean;
  require_budgeted_tag?: boolean;
}

interface Props {
  value: unknown;
  disabled?: boolean;
  onChange: (rules: RequestTagMapping[]) => void;
}

const isMapping = (value: unknown): value is RequestTagMapping => {
  if (typeof value !== "object" || value === null) return false;
  return "key" in value && typeof value.key === "string";
};

const MODES: { value: string; label: ParseKeys }[] = [
  { value: "exact", label: "routerSettings.requestTagMappings.modes.exact" },
  { value: "any", label: "routerSettings.requestTagMappings.modes.any" },
];

const BAGGAGE_KEY_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;

const isExact = (rule: RequestTagMapping) => rule.value !== undefined && rule.value !== null;

const mappingProblem = (rule: RequestTagMapping): ParseKeys | null => {
  if (!rule.key) return "routerSettings.requestTagMappings.problems.keyRequired";
  if (!BAGGAGE_KEY_PATTERN.test(rule.key)) return "routerSettings.requestTagMappings.problems.keyInvalid";
  if (isExact(rule)) {
    if (!rule.value) return "routerSettings.requestTagMappings.problems.valueRequired";
    return rule.tag ? null : "routerSettings.requestTagMappings.problems.tagRequired";
  }
  return rule.tag_prefix ? null : "routerSettings.requestTagMappings.problems.tagPrefixRequired";
};

const matchIdentity = (rule: RequestTagMapping) => JSON.stringify([rule.key, isExact(rule) ? rule.value : null]);

const problemAt = (rules: RequestTagMapping[], index: number): ParseKeys | null => {
  const identity = matchIdentity(rules[index]);
  const duplicated = rules.slice(0, index).some((earlier) => matchIdentity(earlier) === identity);
  return mappingProblem(rules[index]) ?? (duplicated ? "routerSettings.requestTagMappings.problems.duplicate" : null);
};

export const requestTagMappingsAreValid = (value: unknown) =>
  Array.isArray(value) && value.every(isMapping) && value.every((_, index) => problemAt(value, index) === null);

export function RequestTagMappingsEditor({ value, onChange, disabled = false }: Props) {
  const { t } = useTranslation();
  const rules = Array.isArray(value) ? value.filter(isMapping) : [];
  const update = (index: number, patch: Partial<RequestTagMapping>) =>
    onChange(rules.map((rule, position) => (position === index ? { ...rule, ...patch } : rule)));

  return (
    <div className="min-w-80 space-y-4 whitespace-normal">
      <p className="text-xs text-muted-foreground">{t("routerSettings.requestTagMappings.priorityHint")}</p>
      {rules.map((rule, index) => {
        const exact = isExact(rule);
        const problem = problemAt(rules, index);
        const label = t("routerSettings.requestTagMappings.mappingLabel", { number: index + 1 });
        return (
          <fieldset key={index} disabled={disabled} className="space-y-3 rounded-lg border p-3 disabled:opacity-60">
            <legend className="px-1 text-xs font-medium">{label}</legend>
            <div className="flex items-center gap-2">
              <Select
                items={MODES.map((mode) => ({ ...mode, label: t(mode.label) }))}
                value={exact ? "exact" : "any"}
                onValueChange={(mode) =>
                  update(
                    index,
                    mode === "exact"
                      ? { value: "", tag: "", tag_prefix: undefined }
                      : { value: undefined, tag: undefined, tag_prefix: "" },
                  )
                }
              >
                <SelectTrigger aria-label={t("routerSettings.requestTagMappings.modeAriaLabel", { number: index + 1 })}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODES.map((mode) => (
                    <SelectItem key={mode.value} value={mode.value}>
                      {t(mode.label)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={t("routerSettings.requestTagMappings.deleteAriaLabel", { number: index + 1 })}
                onClick={() => onChange(rules.filter((_, position) => position !== index))}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
            <label className="flex flex-col gap-1 text-xs">
              {t("routerSettings.requestTagMappings.baggageKey")}
              <Input
                aria-label={t("routerSettings.requestTagMappings.baggageKeyAriaLabel", { number: index + 1 })}
                placeholder="entrypoint"
                value={rule.key}
                onChange={(event) => update(index, { key: event.target.value })}
              />
            </label>
            {exact ? (
              <>
                <label className="flex flex-col gap-1 text-xs">
                  {t("routerSettings.requestTagMappings.matchValue")}
                  <Input
                    aria-label={t("routerSettings.requestTagMappings.matchValueAriaLabel", { number: index + 1 })}
                    value={rule.value ?? ""}
                    onChange={(event) => update(index, { value: event.target.value })}
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs">
                  {t("routerSettings.requestTagMappings.tag")}
                  <Input
                    aria-label={t("routerSettings.requestTagMappings.tagAriaLabel", { number: index + 1 })}
                    value={rule.tag ?? ""}
                    onChange={(event) => update(index, { tag: event.target.value })}
                  />
                </label>
                <p className="text-xs text-muted-foreground">
                  {rule.key}={rule.value || t("routerSettings.requestTagMappings.previewValue")} →{" "}
                  {rule.tag || t("routerSettings.requestTagMappings.previewTag")}
                </p>
              </>
            ) : (
              <>
                <label className="flex flex-col gap-1 text-xs">
                  {t("routerSettings.requestTagMappings.tagPrefix")}
                  <Input
                    aria-label={t("routerSettings.requestTagMappings.tagPrefixAriaLabel", { number: index + 1 })}
                    placeholder="usecase:"
                    value={rule.tag_prefix ?? ""}
                    onChange={(event) => update(index, { tag_prefix: event.target.value })}
                  />
                </label>
                <p className="text-xs text-muted-foreground">
                  {rule.key}=x → {rule.tag_prefix}x
                </p>
              </>
            )}
            {problem ? <p className="text-xs text-destructive">{t(problem)}</p> : null}
            <label className="flex items-center justify-between gap-3 text-xs">
              {t("routerSettings.requestTagMappings.required")}
              <Switch
                aria-label={t("routerSettings.requestTagMappings.requiredAriaLabel", { number: index + 1 })}
                checked={rule.required ?? false}
                onCheckedChange={(checked) => update(index, { required: checked })}
              />
            </label>
            <label className="flex items-center justify-between gap-3 text-xs">
              {t("routerSettings.requestTagMappings.requireBudget")}
              <Switch
                aria-label={t("routerSettings.requestTagMappings.requireBudgetAriaLabel", { number: index + 1 })}
                checked={rule.require_budgeted_tag ?? false}
                onCheckedChange={(checked) => update(index, { require_budgeted_tag: checked })}
              />
            </label>
          </fieldset>
        );
      })}
      {disabled ? (
        <p className="text-xs text-muted-foreground">{t("routerSettings.requestTagMappings.configManaged")}</p>
      ) : null}
      <Button
        disabled={disabled}
        type="button"
        variant="outline"
        onClick={() => onChange([...rules, { header: "baggage", key: "", value: "", tag: "" }])}
      >
        {t("routerSettings.requestTagMappings.addMapping")}
      </Button>
    </div>
  );
}
