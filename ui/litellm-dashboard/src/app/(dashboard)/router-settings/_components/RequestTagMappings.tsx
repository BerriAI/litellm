import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Trash2 } from "lucide-react";

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

export function RequestTagMappingsEditor({ value, onChange, disabled = false }: Props) {
  const rules = Array.isArray(value) ? value.filter(isMapping) : [];
  const update = (index: number, patch: Partial<RequestTagMapping>) =>
    onChange(rules.map((rule, position) => (position === index ? { ...rule, ...patch } : rule)));

  return (
    <div className="min-w-80 space-y-4">
      <p className="text-xs text-muted-foreground">
        Exact value mappings take priority over any-value rules for the same key.
      </p>
      {rules.map((rule, index) => {
        const exact = rule.value !== undefined && rule.value !== null;
        const label = `Mapping ${index + 1}`;
        return (
          <fieldset key={index} disabled={disabled} className="space-y-3 rounded-lg border p-3 disabled:opacity-60">
            <legend className="px-1 text-xs font-medium">{label}</legend>
            <div className="flex items-center gap-2">
              <Select
                value={exact ? "exact" : "any"}
                onValueChange={(mode) =>
                  update(
                    index,
                    mode === "exact"
                      ? { value: "", tag: "", tag_prefix: undefined }
                      : { value: undefined, tag: undefined, tag_prefix: "usecase:" },
                  )
                }
              >
                <SelectTrigger aria-label={`${label} mode`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="exact">Exact value</SelectItem>
                  <SelectItem value="any">Any value</SelectItem>
                </SelectContent>
              </Select>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={`Delete ${label.toLowerCase()}`}
                onClick={() => onChange(rules.filter((_, position) => position !== index))}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
            <label className="block text-xs">
              Baggage key
              <Input
                aria-label={`${label} baggage key`}
                value={rule.key}
                onChange={(event) => update(index, { key: event.target.value })}
              />
            </label>
            {exact ? (
              <>
                <label className="block text-xs">
                  Match value
                  <Input
                    aria-label={`${label} match value`}
                    value={rule.value ?? ""}
                    onChange={(event) => update(index, { value: event.target.value })}
                  />
                </label>
                <label className="block text-xs">
                  Tag
                  <Input
                    aria-label={`${label} tag`}
                    value={rule.tag ?? ""}
                    onChange={(event) => update(index, { tag: event.target.value })}
                  />
                </label>
                <p className="text-xs text-muted-foreground">
                  {rule.key}={rule.value || "value"} → {rule.tag || "tag"}
                </p>
              </>
            ) : (
              <>
                <label className="block text-xs">
                  Tag prefix
                  <Input
                    aria-label={`${label} tag prefix`}
                    value={rule.tag_prefix ?? ""}
                    onChange={(event) => update(index, { tag_prefix: event.target.value })}
                  />
                </label>
                <p className="text-xs text-muted-foreground">
                  {rule.key}=x → {rule.tag_prefix}x
                </p>
              </>
            )}
            <label className="flex items-center justify-between gap-3 text-xs">
              Require a mapped value
              <Switch
                aria-label={`${label} required`}
                checked={rule.required ?? false}
                onCheckedChange={(checked) => update(index, { required: checked })}
              />
            </label>
            <label className="flex items-center justify-between gap-3 text-xs">
              Require an existing tag budget
              <Switch
                aria-label={`${label} require budget`}
                checked={rule.require_budgeted_tag ?? false}
                onCheckedChange={(checked) => update(index, { require_budgeted_tag: checked })}
              />
            </label>
          </fieldset>
        );
      })}
      {disabled ? (
        <p className="text-xs text-muted-foreground">Managed in config.yaml. Edit the YAML to change these rules.</p>
      ) : null}
      <Button
        disabled={disabled}
        type="button"
        variant="outline"
        onClick={() => onChange([...rules, { header: "baggage", key: "entrypoint", value: "", tag: "" }])}
      >
        Add mapping
      </Button>
    </div>
  );
}
