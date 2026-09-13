import { ChevronDown } from "lucide-react";
import React, { useState } from "react";
import { useComplexityScorerDefaults } from "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { type ComplexityRouterConfigValue, heuristicScoringRole } from "./ComplexityRouterConfig";
import {
  dimensionLabel,
  effectiveDimensionWeights,
  rebalanceDimensionWeights,
  type WeightEdit,
} from "./heuristic_scoring_knobs";
import CustomDimensionRows from "./CustomDimensionRows";
import { customDimensionsError } from "./custom_dimensions";

export type KnobGroup = "tier_boundaries" | "token_thresholds" | "dimension_weights";

interface GroupSpec {
  group: KnobGroup;
  title: string;
  blurb: string;
  min: number;
  max?: number;
  step: number;
  withSlider: boolean;
  labels: Record<string, string>;
}

const OVERRIDE_FLOOR_ID = "reasoning-override-min-score";

const GROUPS: GroupSpec[] = [
  {
    group: "tier_boundaries",
    title: "Tier boundaries",
    blurb:
      "The weighted score each tier starts at. Scores run from -1 to 1, and short or conversational prompts score below 0, so a negative boundary is a valid way to lift trivial traffic into a higher tier.",
    min: -1,
    max: 1,
    step: 0.01,
    withSlider: false,
    labels: {
      simple_medium: "Simple to Medium",
      medium_complex: "Medium to Complex",
      complex_reasoning: "Complex to Reasoning",
    },
  },
  {
    group: "token_thresholds",
    title: "Token thresholds",
    blurb:
      "Estimated prompt length, in tokens, that pushes the token count dimension to its floor or ceiling. Lengths between the two score neutral.",
    min: 0,
    step: 1,
    withSlider: false,
    labels: { simple: "Short below", complex: "Long above" },
  },
  {
    group: "dimension_weights",
    title: "Dimension weights",
    blurb:
      "Changing a weight rebalances the other built-in and custom weights to total 1.00. Save stores those values. Untouched routers keep their existing weights.",
    min: 0,
    max: 1,
    step: 0.01,
    withSlider: true,
    labels: {},
  },
];

/** Why a group is currently misconfigured, or null. Never blocks the save: a router written this way in
 *  config.yaml would otherwise be uneditable here for every unrelated change. */
const warn = (group: KnobGroup, values: Record<string, number>): string | null => {
  if (
    group === "tier_boundaries" &&
    (values.simple_medium > values.medium_complex || values.medium_complex > values.complex_reasoning)
  )
    return "These boundaries decrease, so every tier between them is unreachable and its traffic routes elsewhere.";
  if (group === "token_thresholds" && values.simple >= values.complex)
    return "The short threshold is not below the long one, so no prompt length scores neutral on length.";
  return null;
};

interface HeuristicScoringConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}

const HeuristicScoringConfig: React.FC<HeuristicScoringConfigProps> = ({ value, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [draft, setDraft] = useState<{ id: string; raw: string } | null>(null);
  const { data: defaults, isPending, isError, refetch } = useComplexityScorerDefaults();

  // The panel owns its own visibility: the scorer does not run at all when an LLM classifier
  // falls back to the default model, so there is nothing here to configure.
  const scorerRuns = heuristicScoringRole(value) !== "never";
  const customEnabled = heuristicScoringRole(value) === "decides";
  const customRows = customEnabled ? value.custom_dimensions : undefined;
  const [weightError, setWeightError] = useState<string | null>(null);
  const rowError = customDimensionsError(customRows);
  const changeWeights = (edit: WeightEdit) => {
    const result = rebalanceDimensionWeights(defaults?.dimension_weights, value.dimension_weights, customRows, edit);
    if (!result.ok) {
      setWeightError(result.error);
      return;
    }
    setWeightError(null);
    onChange({ ...value, dimension_weights: result.dimension_weights, custom_dimensions: result.custom_dimensions });
  };

  // What an untouched override floor follows: the boundary in effect, override included, not the shipped one.
  const trackedFloor: number | undefined = { ...defaults?.tier_boundaries, ...value.tier_boundaries }.simple_medium;

  const overrides =
    GROUPS.filter((spec) => value[spec.group] !== undefined).length +
    (value.reasoning_override_min_score !== undefined ? 1 : 0);

  // min/max are inert on a text input, and a plain number input renders Number("0.") as "0" so a decimal
  // cannot be typed. Hence the local draft plus an explicit clamp here.
  const commit = (spec: GroupSpec, effective: Record<string, number>, key: string, raw: string) => {
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    if (spec.group === "dimension_weights") {
      changeWeights({ type: "set", target: { kind: "builtin", id: key }, weight: parsed });
      return;
    }
    const clamped = Math.min(spec.max ?? Infinity, Math.max(spec.min, parsed));
    onChange({
      ...value,
      [spec.group]: { ...effective, [key]: spec.step === 1 ? Math.round(clamped) : clamped },
    });
  };

  const commitOverrideFloor = (raw: string) => {
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    onChange({ ...value, reasoning_override_min_score: Math.min(1, Math.max(-1, parsed)) });
  };

  if (!scorerRuns) return null;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mt-4">
      <CollapsibleTrigger render={<button type="button" className="flex w-full items-center gap-2 text-left" />}>
        <ChevronDown
          className={`size-4 shrink-0 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
        <span className="text-sm font-medium">Advanced scoring</span>
        {overrides > 0 && (
          <Badge variant="secondary" data-testid="advanced-scoring-override-count">
            {overrides} {overrides === 1 ? "override" : "overrides"}
          </Badge>
        )}
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-3 space-y-6 pl-6">
          <p className="text-xs text-muted-foreground">
            Every knob below is optional. Left untouched, the router follows the shipped defaults, so it picks up any
            recalibration of them rather than staying pinned to the numbers shown here.
          </p>

          {isPending ? (
            <p className="text-xs text-muted-foreground">Loading the shipped defaults...</p>
          ) : (
            <>
              {isError && (
                <div className="flex items-start gap-2" role="alert">
                  <p className="text-xs font-medium text-destructive">
                    Could not load the shipped defaults, so only values this router already overrides are shown. Saving
                    still works, and an untouched knob keeps following the defaults.
                  </p>
                  <Button type="button" variant="link" size="xs" onClick={() => void refetch()}>
                    Retry
                  </Button>
                </div>
              )}
              {GROUPS.map((spec) => {
                const shipped = defaults?.[spec.group] ?? value[spec.group] ?? {};
                const effective: Record<string, number> =
                  spec.group === "dimension_weights"
                    ? effectiveDimensionWeights(shipped, value.dimension_weights)
                    : { ...shipped, ...value[spec.group] };
                const total =
                  Object.values(effective).reduce((sum, weight) => sum + weight, 0) +
                  (customRows ?? []).reduce((sum, row) => sum + row.weight, 0);
                const problem = warn(spec.group, effective);
                const overridden =
                  value[spec.group] !== undefined || (spec.withSlider && value.custom_dimensions !== undefined);
                const resettable = spec.withSlider || overridden;
                const scoringError = spec.withSlider ? weightError || rowError : null;
                return (
                  <section key={spec.group} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{spec.title}</span>
                        {/* Only a known dimension set has a meaningful total; summing the overrides
                            alone would state a total that is not the router's. */}
                        {spec.withSlider && defaults !== undefined && (
                          <span className="text-xs text-muted-foreground" data-testid="dimension-weight-total">
                            total {total.toFixed(2)}
                          </span>
                        )}
                      </div>
                      {resettable && (
                        <Button
                          type="button"
                          variant="link"
                          size="xs"
                          disabled={!overridden}
                          onClick={() => {
                            setWeightError(null);
                            onChange({
                              ...value,
                              [spec.group]: undefined,
                              ...(spec.withSlider && { custom_dimensions: undefined }),
                            });
                          }}
                        >
                          {spec.withSlider ? "Restore default weights" : "Reset to defaults"}
                        </Button>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">{spec.blurb}</p>

                    {Object.keys(effective).map((key) => {
                      const id = `${spec.group}-${key}`;
                      const label = spec.labels[key] ?? dimensionLabel(key);
                      return (
                        <div key={key} className="flex items-center gap-3">
                          <Label htmlFor={id} className="w-44 text-xs font-normal">
                            {label}
                          </Label>
                          {spec.withSlider && (
                            <Slider
                              min={spec.min}
                              max={spec.max}
                              step={spec.step}
                              disabled={defaults === undefined}
                              value={[effective[key]]}
                              onValueChange={(next) =>
                                commit(spec, effective, key, String(Array.isArray(next) ? next[0] : next))
                              }
                              className="flex-1"
                              aria-label={`${label} weight`}
                            />
                          )}
                          <Input
                            id={id}
                            type="text"
                            inputMode="decimal"
                            className={spec.withSlider ? "w-24" : "w-28"}
                            disabled={spec.withSlider && defaults === undefined}
                            value={draft?.id === id ? draft.raw : Number(effective[key].toPrecision(6)).toString()}
                            onChange={(event) => {
                              setDraft({ id, raw: event.target.value });
                              commit(spec, effective, key, event.target.value);
                            }}
                            onBlur={() => setDraft(null)}
                          />
                        </div>
                      );
                    })}

                    {spec.withSlider && customEnabled && (
                      <CustomDimensionRows
                        rows={customRows ?? []}
                        disabled={defaults === undefined}
                        onChange={(rows) => onChange({ ...value, custom_dimensions: rows })}
                        onWeight={(id, weight) =>
                          changeWeights({ type: "set", target: { kind: "custom", id }, weight })
                        }
                        onAdd={() =>
                          changeWeights({
                            type: "add",
                            row: { id: crypto.randomUUID(), name: "", weight: 0.1, scoring_mode: "match_count" },
                          })
                        }
                        onRemove={(id) => changeWeights({ type: "remove", id })}
                      />
                    )}
                    {scoringError && (
                      <p className="text-xs text-destructive" role="alert">
                        {scoringError}
                      </p>
                    )}
                    {problem && (
                      <p className="text-xs font-medium text-destructive" role="alert">
                        {problem}
                      </p>
                    )}
                  </section>
                );
              })}

              <section className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Reasoning override floor</span>
                  {value.reasoning_override_min_score !== undefined && (
                    <Button
                      type="button"
                      variant="link"
                      size="xs"
                      onClick={() => onChange({ ...value, reasoning_override_min_score: undefined })}
                    >
                      Reset to defaults
                    </Button>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  Two or more reasoning markers promote a request to the reasoning tier, but only once its weighted
                  score reaches this floor.{" "}
                  {trackedFloor === undefined
                    ? "Left untouched, it tracks the Simple to Medium boundary."
                    : `Left untouched, it tracks the Simple to Medium boundary, currently ${trackedFloor.toFixed(2)}.`}{" "}
                  Set it to 0 to promote on the markers alone.
                </p>
                <div className="flex items-center gap-3">
                  <Label htmlFor={OVERRIDE_FLOOR_ID} className="w-44 text-xs font-normal">
                    Minimum score
                  </Label>
                  <Input
                    id={OVERRIDE_FLOOR_ID}
                    type="text"
                    inputMode="decimal"
                    className="w-28"
                    placeholder={trackedFloor === undefined ? undefined : trackedFloor.toFixed(2)}
                    value={
                      draft?.id === OVERRIDE_FLOOR_ID ? draft.raw : value.reasoning_override_min_score?.toString() ?? ""
                    }
                    onChange={(event) => {
                      setDraft({ id: OVERRIDE_FLOOR_ID, raw: event.target.value });
                      commitOverrideFloor(event.target.value);
                    }}
                    onBlur={() => setDraft(null)}
                  />
                </div>
              </section>
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default HeuristicScoringConfig;
