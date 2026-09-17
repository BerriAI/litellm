import React from "react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronRight } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import FuseProfilePresets from "./FuseProfilePresets";
import { Switch } from "@/components/ui/switch";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { MultiSelect } from "@/components/shared/MultiSelect";
import {
  type ComplexityRouterConfigValue,
  type ClassificationFrequency,
  classificationFrequency,
  withClassificationFrequency,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
} from "./ComplexityRouterConfig";
import {
  forecastTierNames,
  forecastModels,
  getForecastConfigError,
  newCapabilitySettings,
  newFuseSettings,
  type CapabilitySettings,
  type FuseSettings,
} from "./forecast_classifier_config";
import ClassifierReasoningEffortSelect from "./ClassifierReasoningEffortSelect";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import ClassifierVisionConfig from "./ClassifierVisionConfig";
import TierModelEffortRows from "./TierModelEffortRows";
import { activeTierRows } from "./tier_rows";
import { setTierModels } from "./tier_set_actions";
import { tierRowLabel, setTierModelParam, setTierModelReasoningEffort } from "./complexity_router_tiers";

interface Props {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  effortOptionsByModel: Record<string, string[] | null | undefined>;
}

const NumberField = ({
  label,
  value,
  onChange,
  min,
  max,
  step = "any",
  help,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number | "any";
  help?: string;
}) => {
  const id = React.useId();
  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number.isFinite(value) ? value : ""}
        onChange={(event) => onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))}
      />
      {help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  );
};

export const ForecastSolverModels = ({
  value,
  onChange,
  modelOptions,
  effortOptionsByModel,
  fastModeByModel,
  additionalPoolsOnly = false,
}: Props & { fastModeByModel: Record<string, boolean>; additionalPoolsOnly?: boolean }) => {
  const id = React.useId();
  const names = forecastTierNames(value);
  const additionalRows =
    value.classifier_type === "capability"
      ? activeTierRows(value)
          .filter((row) => !names.includes(row.id) && row.models.length > 0)
          .map((row) => ({ tier: row.id, label: `${tierRowLabel(row, value.tier_labels)} routing pool` }))
      : [];
  const rows = additionalPoolsOnly
    ? additionalRows
    : names.map((tier, index) => ({ tier, label: index === 0 ? "Efficient solver" : "Capable solver" }));
  if (rows.length === 0) return null;
  return (
    <div className="rounded-lg border p-4 space-y-4">
      {rows.map(({ tier, label }) => {
        const models = forecastModels(value.tiers, tier);
        const setModels = (next: string[]) => onChange(setTierModels(value, tier, next));
        return (
          <div key={tier} className="space-y-2">
            <Label htmlFor={`${id}-${tier}`} className="block text-sm font-semibold">
              {label}
            </Label>
            {value.classifier_type === "llm_v2" ? (
              <SearchSelect
                options={modelOptions}
                inputId={`${id}-${tier}`}
                value={models[0] ?? ""}
                aria-label={label}
                placeholder={`Select ${label.toLowerCase()}`}
                onValueChange={(model) => setModels(model ? [model] : [])}
              />
            ) : (
              <MultiSelect
                options={modelOptions}
                id={`${id}-${tier}`}
                value={models}
                onValueChange={setModels}
                placeholder={`Select ${label.toLowerCase()} models`}
              />
            )}
            <TierModelEffortRows
              tierLabel={label}
              models={models}
              effortOptionsByModel={Object.fromEntries(
                Object.entries(effortOptionsByModel).map(([model, efforts]) => [model, efforts ?? []]),
              )}
              paramsByModel={value.tier_model_params?.[tier] ?? {}}
              fastModeByModel={fastModeByModel}
              onFastModeChange={(model, enabled) =>
                onChange({
                  ...value,
                  tier_model_params: setTierModelParam(value.tier_model_params, tier, model, [
                    "speed",
                    enabled ? "fast" : undefined,
                  ]),
                })
              }
              onEffortChange={(model, effort) =>
                onChange({
                  ...value,
                  tier_model_params: setTierModelReasoningEffort(value.tier_model_params, tier, model, effort),
                })
              }
            />
          </div>
        );
      })}
      {!additionalPoolsOnly && (
        <p className="text-sm text-muted-foreground">
          Invalid forecasts and classifier failures route to the capable solver
        </p>
      )}
    </div>
  );
};

const CalibrationFields = ({
  label,
  value,
  onChange,
  bounded = false,
}: {
  label: string;
  bounded?: boolean;
  value: { slope: number; intercept: number };
  onChange: (value: { slope: number; intercept: number }) => void;
}) => (
  <div className="grid gap-3 sm:grid-cols-2">
    <NumberField
      label={`${label} slope`}
      value={value.slope}
      min={0}
      max={bounded ? 20 : undefined}
      onChange={(slope) => onChange({ ...value, slope })}
    />
    <NumberField
      label={`${label} intercept`}
      value={value.intercept}
      min={bounded ? -20 : undefined}
      max={bounded ? 20 : undefined}
      onChange={(intercept) => onChange({ ...value, intercept })}
    />
  </div>
);

const emptyCoefficients = () => ({ slope: Number.NaN, intercept: Number.NaN });

const ForecastClassifierConfig = ({ value, onChange, modelOptions, effortOptionsByModel }: Props) => {
  const id = React.useId();
  const isCapability = value.classifier_type === "capability";
  const capability = value.capability_classifier_config ?? newCapabilitySettings();
  const fuse = value.llm_v2_config ?? newFuseSettings();
  const config = isCapability ? capability : fuse;
  const llm = value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS };
  const updateCapability = (next: CapabilitySettings) => onChange({ ...value, capability_classifier_config: next });
  const updateFuse = (next: FuseSettings) => onChange({ ...value, llm_v2_config: next });
  const updateTransport = (patch: { max_output_tokens?: number; response_format?: "json_schema" | "json_object" }) =>
    isCapability ? updateCapability({ ...capability, ...patch }) : updateFuse({ ...fuse, ...patch });
  const setCalibrationVersion = (version: string) => {
    if (isCapability && capability.calibration)
      updateCapability({ ...capability, calibration: { ...capability.calibration, version } });
    if (!isCapability && fuse.calibration) updateFuse({ ...fuse, calibration: { ...fuse.calibration, version } });
  };
  const error = getForecastConfigError(value);
  return (
    <div className="mt-4 space-y-4">
      <p className="text-sm text-muted-foreground">
        {isCapability
          ? "Forecasts whether the efficient solver can complete the task using the bundled capability card"
          : "Forecasts success for both solvers and selects efficient when the estimated quality gap is within your allowance"}
      </p>
      <div className="space-y-1">
        <Label htmlFor={`${id}-judge`}>Judge model</Label>
        <SearchSelect
          inputId={`${id}-judge`}
          aria-label="Judge model"
          options={modelOptions}
          value={llm.model}
          placeholder="Select the judge model"
          onValueChange={(model) => {
            if (model === llm.model) return;
            onChange({ ...value, classifier_llm_config: { ...llm, model: model ?? "", reasoning_effort: undefined } });
          }}
        />
      </div>
      {isCapability ? (
        <>
          <NumberField
            label="Solve probability threshold"
            value={capability.base_threshold}
            min={0}
            max={1}
            help="Minimum estimated chance of whole-task success required to use the efficient solver"
            onChange={(base_threshold) => updateCapability({ ...capability, base_threshold })}
          />
        </>
      ) : (
        <>
          <FuseProfilePresets value={fuse} onChange={updateFuse} />
          <NumberField
            label="Maximum quality gap"
            value={fuse.max_quality_gap}
            min={0}
            max={1}
            help="Allowed difference between capable and efficient success probabilities, from 0 to 1. This is an estimate, not a measured quality guarantee"
            onChange={(max_quality_gap) => updateFuse({ ...fuse, max_quality_gap })}
          />
        </>
      )}
      <Collapsible className="rounded-lg border">
        <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left font-medium">
          <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" />
          Classifier options
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 px-4 pb-4">
          <ClassifierReasoningEffortSelect
            model={llm.model}
            value={llm.reasoning_effort}
            explicitlySupported={effortOptionsByModel[llm.model]}
            onChange={(reasoning_effort) => onChange({ ...value, classifier_llm_config: { ...llm, reasoning_effort } })}
          />
          <NumberField
            label="Timeout (ms)"
            min={1}
            step={1}
            value={llm.timeout_ms}
            help="Allow enough time for the judge to produce its forecast"
            onChange={(timeout_ms) => onChange({ ...value, classifier_llm_config: { ...llm, timeout_ms } })}
          />
          <ClassifierCircuitBreakerConfig
            value={llm}
            onChange={(classifier_llm_config) => onChange({ ...value, classifier_llm_config })}
          />
          <ClassifierVisionConfig
            value={llm}
            onChange={(classifier_llm_config) => onChange({ ...value, classifier_llm_config })}
          />
          <div className="space-y-1">
            <Label htmlFor={`${id}-frequency`}>How often to classify</Label>
            <SearchSelect
              inputId={`${id}-frequency`}
              aria-label="How often to classify"
              value={classificationFrequency(value)}
              allowClear={false}
              options={[
                { value: "every_request", label: "Every request" },
                { value: "user_turn", label: "Every new user message" },
                { value: "session", label: "Once per session" },
              ]}
              onValueChange={(frequency) => {
                if (frequency) onChange(withClassificationFrequency(value, frequency as ClassificationFrequency));
              }}
            />
          </div>
          {isCapability && (
            <NumberField
              label="Capability boundary step"
              value={capability.threshold_step ?? 0}
              min={0}
              max={0.5}
              help="Added once for uncertain or unmatched tasks and twice for unsupported tasks; the final threshold cannot exceed 1"
              onChange={(threshold_step) => updateCapability({ ...capability, threshold_step })}
            />
          )}
          <NumberField
            label="Classifier output token limit"
            min={1}
            step={1}
            value={config.max_output_tokens ?? (isCapability ? 4096 : 1024)}
            onChange={(max_output_tokens) => updateTransport({ max_output_tokens })}
          />
          <div className="space-y-1">
            <Label htmlFor={`${id}-format`}>Forecast response format</Label>
            <SearchSelect
              inputId={`${id}-format`}
              aria-label="Forecast response format"
              value={config.response_format ?? "json_schema"}
              allowClear={false}
              options={[
                { value: "json_schema", label: "Strict JSON schema" },
                { value: "json_object", label: "JSON object (for judges without strict schema support)" },
              ]}
              onValueChange={(response_format) => {
                if (response_format === "json_schema" || response_format === "json_object")
                  updateTransport({ response_format });
              }}
            />
          </div>
          <div className="space-y-3 rounded-md border p-3">
            <Label>
              <Switch
                checked={Boolean(config.calibration)}
                onCheckedChange={(enabled) =>
                  isCapability
                    ? updateCapability({
                        ...capability,
                        calibration: enabled ? { version: "", ...emptyCoefficients() } : undefined,
                      })
                    : updateFuse({
                        ...fuse,
                        calibration: enabled
                          ? {
                              version: "",
                              prompt_version: "llm-v2-1",
                              efficient: emptyCoefficients(),
                              capable: emptyCoefficients(),
                            }
                          : undefined,
                      })
                }
              />
              Use fitted calibration
            </Label>
            <p className="text-xs text-muted-foreground">
              Optional coefficients fitted for your judge, solvers, and harness. Leave off to use raw forecasts
            </p>
            {config.calibration && (
              <div className="space-y-1">
                <Label htmlFor={`${id}-version`}>Calibration version</Label>
                <Input
                  id={`${id}-version`}
                  value={config.calibration.version}
                  maxLength={isCapability ? 128 : 512}
                  onChange={(event) => setCalibrationVersion(event.target.value)}
                />
              </div>
            )}
            {isCapability && capability.calibration && (
              <CalibrationFields
                label="Efficient"
                bounded
                value={capability.calibration}
                onChange={(next) =>
                  updateCapability({
                    ...capability,
                    calibration: { version: capability.calibration?.version ?? "", ...next },
                  })
                }
              />
            )}
            {!isCapability &&
              fuse.calibration &&
              (["efficient", "capable"] as const).map((role) => (
                <CalibrationFields
                  key={role}
                  label={role === "efficient" ? "Efficient" : "Capable"}
                  value={fuse.calibration![role]}
                  onChange={(next) => {
                    if (fuse.calibration) updateFuse({ ...fuse, calibration: { ...fuse.calibration, [role]: next } });
                  }}
                />
              ))}
          </div>
        </CollapsibleContent>
      </Collapsible>
      <p className="text-xs text-muted-foreground">
        The classifier uses its bundled prompt and always falls back to the capable solver
      </p>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
};

export default ForecastClassifierConfig;
