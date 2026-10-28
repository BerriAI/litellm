import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import React from "react";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const OUTPUT_TOKEN_POLICIES = [
  {
    value: true,
    label: "Use model ceiling",
    description: "Replace the caller's output limit with the selected model's ceiling. This is the default.",
  },
  {
    value: "clamp",
    label: "Respect caller budget",
    description:
      "Use the smaller of the caller's output limit and the model ceiling. Use the ceiling when no limit is sent.",
  },
  {
    value: false,
    label: "Pass through caller limit",
    description: "Forward the caller's output limit unchanged.",
  },
] as const;

const ResponseFormatControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.return_raw_model_name ?? false}
        onCheckedChange={(returnRawModelName) => onChange({ ...value, return_raw_model_name: returnRawModelName })}
        aria-label="Return raw model name"
      />
      <strong className="font-semibold">Return raw model name</strong>
    </div>
    <span className="block text-xs text-muted-foreground">
      Return the resolved underlying model name in responses instead of the autorouter alias.
    </span>
    <div className="mt-4 space-y-2">
      <strong className="font-semibold">Output token limit</strong>
      <RadioGroup
        value={value.max_tokens_from_tier_model ?? true}
        onValueChange={(mode: unknown) => {
          if (typeof mode === "boolean" || mode === "clamp") {
            onChange({ ...value, max_tokens_from_tier_model: mode });
          }
        }}
        aria-label="Output token limit"
      >
        {OUTPUT_TOKEN_POLICIES.map((policy) => (
          <Label key={String(policy.value)} className="items-start font-normal leading-normal">
            <RadioGroupItem value={policy.value} className="mt-0.5" />
            <span>
              <span className="block font-medium">{policy.label}</span>
              <span className="block text-xs text-muted-foreground">{policy.description}</span>
            </span>
          </Label>
        ))}
      </RadioGroup>
      <span className="block text-xs text-muted-foreground">
        Explicit tier output limits take precedence. Unknown model ceilings keep the caller&apos;s limit.
      </span>
    </div>
  </>
);

export default ResponseFormatControls;
