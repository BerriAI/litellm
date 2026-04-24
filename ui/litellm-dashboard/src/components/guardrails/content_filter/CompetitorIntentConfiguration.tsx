import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import { getMajorAirlines } from "../../networking";

export interface MajorAirline {
  id: string;
  match: string;
  tags: string[];
}

export interface CompetitorIntentConfig {
  competitor_intent_type: "airline" | "generic";
  brand_self: string[];
  locations?: string[];
  competitors?: string[];
  policy?: {
    competitor_comparison?: "refuse" | "reframe";
    possible_competitor_comparison?: "refuse" | "reframe";
  };
  threshold_high?: number;
  threshold_medium?: number;
  threshold_low?: number;
}

interface CompetitorIntentConfigurationProps {
  enabled: boolean;
  config: CompetitorIntentConfig | null;
  onChange: (enabled: boolean, config: CompetitorIntentConfig | null) => void;
  accessToken?: string | null;
}

const DEFAULT_CONFIG: CompetitorIntentConfig = {
  competitor_intent_type: "airline",
  brand_self: [],
  locations: [],
  policy: {
    competitor_comparison: "refuse",
    possible_competitor_comparison: "reframe",
  },
  threshold_high: 0.7,
  threshold_medium: 0.45,
  threshold_low: 0.3,
};

/**
 * Simple chip-style tag input used by the competitor-intent form. Mirrors
 * the behavior of antd's `Select mode="tags"` but without using antd.
 */
function TagInput({
  value,
  onChange,
  placeholder,
  disabled = false,
}: {
  value: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState("");

  const commit = () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    const parts = trimmed
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const next = [...value];
    for (const p of parts) {
      if (!next.includes(p)) next.push(p);
    }
    onChange(next);
    setDraft("");
  };

  return (
    <div className="space-y-2">
      <Input
        value={draft}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            commit();
          } else if (e.key === "Backspace" && !draft && value.length > 0) {
            onChange(value.slice(0, -1));
          }
        }}
        onBlur={commit}
      />
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(value.filter((x) => x !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

const CompetitorIntentConfiguration: React.FC<
  CompetitorIntentConfigurationProps
> = ({ enabled, config, onChange, accessToken }) => {
  const effectiveConfig = config ?? DEFAULT_CONFIG;
  const [airlineOptions, setAirlineOptions] = useState<MajorAirline[]>([]);
  const [loadingAirlines, setLoadingAirlines] = useState(false);

  useEffect(() => {
    if (
      effectiveConfig.competitor_intent_type === "airline" &&
      accessToken &&
      airlineOptions.length === 0
    ) {
      setLoadingAirlines(true);
      getMajorAirlines(accessToken)
        .then((res) => setAirlineOptions(res.airlines ?? []))
        .catch(() => setAirlineOptions([]))
        .finally(() => setLoadingAirlines(false));
    }
  }, [effectiveConfig.competitor_intent_type, accessToken, airlineOptions.length]);

  const handleEnabledChange = (checked: boolean) => {
    onChange(checked, checked ? { ...DEFAULT_CONFIG } : null);
  };

  const handleConfigChange = (field: string, value: unknown) => {
    onChange(enabled, { ...effectiveConfig, [field]: value });
  };

  const handlePolicyChange = (key: string, value: string) => {
    onChange(enabled, {
      ...effectiveConfig,
      policy: { ...effectiveConfig.policy, [key]: value },
    });
  };

  const handleNestedArrayChange = (
    field: "brand_self" | "locations" | "competitors",
    values: string[],
  ) => {
    onChange(enabled, { ...effectiveConfig, [field]: values.filter(Boolean) });
  };

  const handleBrandSelfChange = (values: string[]) => {
    const filtered = values.filter(Boolean);
    const expanded: string[] = [];
    const seen = new Set<string>();
    for (const v of filtered) {
      const airline = airlineOptions.find((a) => {
        const primary = a.match.split("|")[0]?.trim().toLowerCase();
        return primary === v.toLowerCase();
      });
      if (airline) {
        for (const variant of airline.match
          .split("|")
          .map((s) => s.trim().toLowerCase())
          .filter(Boolean)) {
          if (!seen.has(variant)) {
            seen.add(variant);
            expanded.push(variant);
          }
        }
      } else if (!seen.has(v.toLowerCase())) {
        seen.add(v.toLowerCase());
        expanded.push(v);
      }
    }
    onChange(enabled, { ...effectiveConfig, brand_self: expanded });
  };

  const header = (
    <div className="flex justify-between items-center">
      <h5 className="text-base font-semibold m-0">Competitor Intent Filter</h5>
      <Switch checked={enabled} onCheckedChange={handleEnabledChange} />
    </div>
  );

  if (!enabled) {
    return (
      <Card className="p-4 space-y-3">
        {header}
        <p className="text-sm text-muted-foreground m-0">
          Block or reframe competitor comparison questions. When enabled, airline
          type auto-loads competitors from IATA; generic type requires manual
          competitor list.
        </p>
      </Card>
    );
  }

  const isAirline = effectiveConfig.competitor_intent_type === "airline";
  const useAirlineSelect = isAirline && airlineOptions.length > 0;

  return (
    <Card className="p-4 space-y-4">
      {header}
      <p className="text-sm text-muted-foreground m-0">
        Block or reframe competitor comparison questions. Airline type uses
        major airlines (excluding your brand); generic requires manual
        competitor list.
      </p>

      <div className="space-y-2">
        <Label>Type</Label>
        <Select
          value={effectiveConfig.competitor_intent_type}
          onValueChange={(v) => handleConfigChange("competitor_intent_type", v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="airline">
              Airline (auto-load competitors from IATA)
            </SelectItem>
            <SelectItem value="generic">
              Generic (specify competitors manually)
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>
          Your Brand (brand_self) <span className="text-destructive">*</span>
        </Label>
        {useAirlineSelect ? (
          <AirlineBrandSelect
            selected={effectiveConfig.brand_self}
            options={airlineOptions}
            loading={loadingAirlines}
            onChange={(values) => handleBrandSelfChange(values)}
          />
        ) : (
          <TagInput
            value={effectiveConfig.brand_self ?? []}
            onChange={(v) => handleNestedArrayChange("brand_self", v)}
            placeholder={
              loadingAirlines
                ? "Loading airlines..."
                : "Type and press Enter to add"
            }
          />
        )}
        <p className="text-xs text-muted-foreground">
          {isAirline
            ? "Select your airline from the list (excluded from competitors) or type to add a custom term"
            : "Names/codes users use for your brand"}
        </p>
      </div>

      {isAirline && (
        <div className="space-y-2">
          <Label>Locations (optional)</Label>
          <TagInput
            value={effectiveConfig.locations ?? []}
            onChange={(v) => handleNestedArrayChange("locations", v)}
            placeholder="Type and press Enter to add"
          />
          <p className="text-xs text-muted-foreground">
            Countries, cities, airports for disambiguation (e.g. qatar, doha)
          </p>
        </div>
      )}

      {effectiveConfig.competitor_intent_type === "generic" && (
        <div className="space-y-2">
          <Label>
            Competitors <span className="text-destructive">*</span>
          </Label>
          <TagInput
            value={effectiveConfig.competitors ?? []}
            onChange={(v) => handleNestedArrayChange("competitors", v)}
            placeholder="Type and press Enter to add"
          />
          <p className="text-xs text-muted-foreground">
            Competitor names to detect (required for generic type)
          </p>
        </div>
      )}

      <div className="space-y-2">
        <Label>Policy: Competitor comparison</Label>
        <Select
          value={effectiveConfig.policy?.competitor_comparison ?? "refuse"}
          onValueChange={(v) => handlePolicyChange("competitor_comparison", v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="refuse">Refuse (block request)</SelectItem>
            <SelectItem value="reframe">Reframe (suggest alternative)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Policy: Possible competitor comparison</Label>
        <Select
          value={
            effectiveConfig.policy?.possible_competitor_comparison ?? "reframe"
          }
          onValueChange={(v) =>
            handlePolicyChange("possible_competitor_comparison", v)
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="refuse">Refuse (block request)</SelectItem>
            <SelectItem value="reframe">
              Reframe (suggest alternative to backend LLM)
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Confidence thresholds</Label>
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <Label htmlFor="threshold-high" className="text-xs">
              High
            </Label>
            <Input
              id="threshold-high"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={effectiveConfig.threshold_high ?? 0.7}
              onChange={(e) =>
                handleConfigChange(
                  "threshold_high",
                  e.target.value === "" ? 0.7 : Number(e.target.value),
                )
              }
              className="w-24"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="threshold-medium" className="text-xs">
              Medium
            </Label>
            <Input
              id="threshold-medium"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={effectiveConfig.threshold_medium ?? 0.45}
              onChange={(e) =>
                handleConfigChange(
                  "threshold_medium",
                  e.target.value === "" ? 0.45 : Number(e.target.value),
                )
              }
              className="w-24"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="threshold-low" className="text-xs">
              Low
            </Label>
            <Input
              id="threshold-low"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={effectiveConfig.threshold_low ?? 0.3}
              onChange={(e) =>
                handleConfigChange(
                  "threshold_low",
                  e.target.value === "" ? 0.3 : Number(e.target.value),
                )
              }
              className="w-24"
            />
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          Classify competitor intent by confidence (0–1). Higher confidence →
          stronger intent.
          <ul className="mt-1 mb-0 pl-5 list-disc">
            <li>
              <strong>High (≥)</strong>: Treat as full competitor comparison →
              uses &quot;Competitor comparison&quot; policy
            </li>
            <li>
              <strong>Medium (≥)</strong>: Treat as possible comparison → uses
              &quot;Possible competitor comparison&quot; policy
            </li>
            <li>
              <strong>Low (≥)</strong>: Log only; allow request. Below Low →
              allow with no action
            </li>
          </ul>
          Raise thresholds to be more permissive; lower them to be stricter.
        </div>
      </div>
    </Card>
  );
};

/**
 * Select + chip display for airlines. Adds on pick; supports custom entries
 * via the parallel TagInput below.
 */
function AirlineBrandSelect({
  selected,
  options,
  loading,
  onChange,
}: {
  selected: string[];
  options: MajorAirline[];
  loading: boolean;
  onChange: (values: string[]) => void;
}) {
  const selectedSet = new Set(selected.map((s) => s.toLowerCase()));
  const unselected = options.filter((a) => {
    const primary = a.match.split("|")[0]?.trim().toLowerCase();
    return primary && !selectedSet.has(primary);
  });

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Select
          value=""
          onValueChange={(v) => {
            if (!v) return;
            onChange([...selected, v]);
          }}
        >
          <SelectTrigger>
            <SelectValue
              placeholder={
                loading ? "Loading airlines..." : "Select airline or pick one"
              }
            />
          </SelectTrigger>
          <SelectContent>
            {unselected.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No airlines available
              </div>
            ) : (
              unselected.map((a) => {
                const primary =
                  a.match.split("|")[0]?.trim().toLowerCase() ?? a.id;
                const label = a.match.split("|")[0]?.trim() ?? a.id;
                const variants = a.match
                  .split("|")
                  .map((s) => s.trim())
                  .filter(Boolean);
                return (
                  <SelectItem key={a.id} value={primary}>
                    {label}
                    {variants.length > 1
                      ? ` (${variants.slice(1).join(", ")})`
                      : ""}
                  </SelectItem>
                );
              })
            )}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-wrap gap-1">
        {selected.map((v) => (
          <Badge
            key={v}
            variant="secondary"
            className="flex items-center gap-1"
          >
            {v}
            <button
              type="button"
              onClick={() => onChange(selected.filter((s) => s !== v))}
              className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
              aria-label={`Remove ${v}`}
            >
              <X size={12} />
            </button>
          </Badge>
        ))}
      </div>
    </div>
  );
}

export default CompetitorIntentConfiguration;
