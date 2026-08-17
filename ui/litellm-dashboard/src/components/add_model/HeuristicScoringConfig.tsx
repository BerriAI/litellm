import { ChevronDown } from "lucide-react";
import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  ComplexityRouterConfigValue,
  DEFAULT_DIMENSION_WEIGHTS,
  DEFAULT_TIER_BOUNDARIES,
  DEFAULT_TOKEN_THRESHOLDS,
  DIMENSION_KEYS,
  DIMENSION_LABELS,
  DimensionKey,
  TierBoundaries,
  TokenThresholds,
} from "./ComplexityRouterConfig";
import { weightTotal } from "./heuristic_scoring_knobs";

const BOUNDARY_ROWS: { key: keyof TierBoundaries; label: string }[] = [
  { key: "simple_medium", label: "Simple to Medium" },
  { key: "medium_complex", label: "Medium to Complex" },
  { key: "complex_reasoning", label: "Complex to Reasoning" },
];

const THRESHOLD_ROWS: { key: keyof TokenThresholds; label: string }[] = [
  { key: "simple", label: "Short below" },
  { key: "complex", label: "Long above" },
];

interface NumberFieldProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max?: number;
  step: number;
  width: string;
  onCommit: (next: number) => void;
  /** Rendered between the label and the input, so a weight row can put its slider there. */
  slot?: React.ReactNode;
}

/**
 * A decimal field cannot be a plain controlled number input: typing "0.22" round-trips through
 * Number("0.") on the second keystroke, which renders as "0" and eats the point. So the raw text is held
 * locally while the field is being edited, only parseable values are committed, and blur drops the draft so
 * the field snaps back to whatever the config actually holds. An emptied field commits nothing.
 */
const NumberField: React.FC<NumberFieldProps> = ({ id, label, value, min, max, step, width, onCommit, slot }) => {
  const [draft, setDraft] = useState<string | null>(null);

  const handleDraftChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setDraft(raw);
    const parsed = Number(raw);
    if (raw.trim() !== "" && !Number.isNaN(parsed)) onCommit(parsed);
  };

  return (
    <div className="flex items-center gap-3">
      <Label htmlFor={id} className="w-44 text-xs font-normal">
        {label}
      </Label>
      {slot}
      <Input
        id={id}
        type="text"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        className={width}
        value={draft ?? String(value)}
        onChange={handleDraftChange}
        onBlur={() => setDraft(null)}
      />
    </div>
  );
};

interface HeuristicScoringConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}

const HeuristicScoringConfig: React.FC<HeuristicScoringConfigProps> = ({ value, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);

  const boundaries = value.tier_boundaries ?? DEFAULT_TIER_BOUNDARIES;
  const thresholds = value.token_thresholds ?? DEFAULT_TOKEN_THRESHOLDS;
  const weights = value.dimension_weights ?? DEFAULT_DIMENSION_WEIGHTS;

  const overrideCount = [value.tier_boundaries, value.token_thresholds, value.dimension_weights].filter(
    (knob) => knob !== undefined,
  ).length;

  const boundariesOutOfOrder =
    boundaries.simple_medium > boundaries.medium_complex || boundaries.medium_complex > boundaries.complex_reasoning;
  const thresholdsOutOfOrder = thresholds.simple >= thresholds.complex;
  const total = weightTotal(weights);
  const totalOffTarget = Math.abs(total - 1) > 0.005;

  const handleBoundaryChange = (key: keyof TierBoundaries, next: number) =>
    onChange({ ...value, tier_boundaries: { ...boundaries, [key]: next } });

  const handleThresholdChange = (key: keyof TokenThresholds, next: number) =>
    onChange({ ...value, token_thresholds: { ...thresholds, [key]: Math.round(next) } });

  const handleWeightChange = (key: DimensionKey, next: number) =>
    onChange({ ...value, dimension_weights: { ...weights, [key]: next } });

  const resetButton = (label: string, reset: () => void) => (
    <Button type="button" variant="link" size="xs" onClick={reset}>
      {label}
    </Button>
  );

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mt-4">
      <CollapsibleTrigger render={<button type="button" className="flex w-full items-center gap-2 text-left" />}>
        <ChevronDown
          className={`size-4 shrink-0 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
        <span className="text-sm font-medium">Advanced scoring</span>
        {overrideCount > 0 && (
          <Badge variant="secondary" data-testid="advanced-scoring-override-count">
            {overrideCount} {overrideCount === 1 ? "override" : "overrides"}
          </Badge>
        )}
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-3 space-y-6 pl-6">
          <p className="text-xs text-muted-foreground">
            Every knob below is optional. Left untouched, the router follows the shipped defaults, so it picks up any
            recalibration of them rather than staying pinned to the numbers shown here.
          </p>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Tier boundaries</span>
              {value.tier_boundaries !== undefined &&
                resetButton("Reset to defaults", () => onChange({ ...value, tier_boundaries: undefined }))}
            </div>
            <p className="text-xs text-muted-foreground">
              The weighted score each tier starts at. Scores run from -1 to 1, and short or conversational prompts score
              below 0, so a negative boundary is a valid way to lift trivial traffic into a higher tier.
            </p>
            {BOUNDARY_ROWS.map(({ key, label }) => (
              <NumberField
                key={key}
                id={`boundary-${key}`}
                label={label}
                value={boundaries[key]}
                min={-1}
                max={1}
                step={0.01}
                width="w-28"
                onCommit={(next) => handleBoundaryChange(key, next)}
              />
            ))}
            {boundariesOutOfOrder && (
              <p className="text-xs text-amber-600">
                Boundaries are out of order, so the tiers in between can never be selected.
              </p>
            )}
          </section>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Token thresholds</span>
              {value.token_thresholds !== undefined &&
                resetButton("Reset to defaults", () => onChange({ ...value, token_thresholds: undefined }))}
            </div>
            <p className="text-xs text-muted-foreground">
              Estimated prompt length, in tokens, that pushes the token count dimension to its floor or ceiling. Lengths
              between the two score neutral.
            </p>
            {THRESHOLD_ROWS.map(({ key, label }) => (
              <NumberField
                key={key}
                id={`threshold-${key}`}
                label={label}
                value={thresholds[key]}
                min={0}
                step={1}
                width="w-28"
                onCommit={(next) => handleThresholdChange(key, next)}
              />
            ))}
            {thresholdsOutOfOrder && (
              <p className="text-xs text-amber-600">
                The short threshold is not below the long one, so no prompt length scores neutral.
              </p>
            )}
          </section>

          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Dimension weights</span>
                <span className="text-xs text-muted-foreground" data-testid="dimension-weight-total">
                  total {total.toFixed(2)}
                </span>
              </div>
              {value.dimension_weights !== undefined &&
                resetButton("Reset to defaults", () => onChange({ ...value, dimension_weights: undefined }))}
            </div>
            {DIMENSION_KEYS.map((key) => (
              <NumberField
                key={key}
                id={`weight-${key}`}
                label={DIMENSION_LABELS[key]}
                value={weights[key]}
                min={0}
                max={1}
                step={0.01}
                width="w-24"
                onCommit={(next) => handleWeightChange(key, next)}
                slot={
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={[weights[key]]}
                    onValueChange={(next) => {
                      const raw = Array.isArray(next) ? next[0] : next;
                      handleWeightChange(key, Math.round(raw * 100) / 100);
                    }}
                    className="flex-1"
                    aria-label={`${DIMENSION_LABELS[key]} weight`}
                  />
                }
              />
            ))}
            {totalOffTarget && (
              <p className="text-xs text-muted-foreground">
                Weights are absolute multipliers, so a total other than 1.00 is accepted. It rescales every score, which
                moves scores relative to the tier boundaries above.
              </p>
            )}
          </section>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default HeuristicScoringConfig;
