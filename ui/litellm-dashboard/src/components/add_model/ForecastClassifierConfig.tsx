import React from "react";
import { useTranslation } from "react-i18next";
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
  section?: "all" | "required" | "advanced";
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
  const { t } = useTranslation();
  const id = React.useId();
  const names = forecastTierNames(value);
  const additionalRows =
    value.classifier_type === "capability"
      ? activeTierRows(value)
          .filter((row) => !names.includes(row.id) && row.models.length > 0)
          .map((row) => ({
            tier: row.id,
            label: t("addModel.forecastClassifierConfig.solvers.routingPool", {
              tier: tierRowLabel(row, value.tier_labels),
            }),
          }))
      : [];
  const rows = additionalPoolsOnly
    ? additionalRows
    : names.map((tier, index) => ({
        tier,
        label:
          index === 0
            ? t("addModel.forecastClassifierConfig.solvers.efficient")
            : t("addModel.forecastClassifierConfig.solvers.capable"),
      }));
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
                placeholder={t("addModel.forecastClassifierConfig.solvers.selectOne", { label: label.toLowerCase() })}
                onValueChange={(model) => setModels(model ? [model] : [])}
              />
            ) : (
              <MultiSelect
                options={modelOptions}
                id={`${id}-${tier}`}
                value={models}
                onValueChange={setModels}
                placeholder={t("addModel.forecastClassifierConfig.solvers.selectModels", {
                  label: label.toLowerCase(),
                })}
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
        <p className="text-sm text-muted-foreground">{t("addModel.forecastClassifierConfig.solvers.failureRouting")}</p>
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
}) => {
  const { t } = useTranslation();
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <NumberField
        label={t("addModel.forecastClassifierConfig.calibrationFields.slope", { label })}
        value={value.slope}
        min={0}
        max={bounded ? 20 : undefined}
        onChange={(slope) => onChange({ ...value, slope })}
      />
      <NumberField
        label={t("addModel.forecastClassifierConfig.calibrationFields.intercept", { label })}
        value={value.intercept}
        min={bounded ? -20 : undefined}
        max={bounded ? 20 : undefined}
        onChange={(intercept) => onChange({ ...value, intercept })}
      />
    </div>
  );
};

const emptyCoefficients = () => ({ slope: Number.NaN, intercept: Number.NaN });

const ForecastClassifierConfig = ({ value, onChange, modelOptions, effortOptionsByModel, section = "all" }: Props) => {
  const { t } = useTranslation();
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
  const error = getForecastConfigError(value, t);
  return (
    <div className="mt-4 space-y-4">
      <p className="text-sm text-muted-foreground">
        {isCapability
          ? t("addModel.forecastClassifierConfig.description.capability")
          : t("addModel.forecastClassifierConfig.description.fuse")}
      </p>
      {section === "all" && (
        <>
          <div className="space-y-1">
            <Label htmlFor={`${id}-judge`}>{t("addModel.forecastClassifierConfig.judge.label")}</Label>
            <SearchSelect
              inputId={`${id}-judge`}
              aria-label={t("addModel.forecastClassifierConfig.judge.label")}
              options={modelOptions}
              value={llm.model}
              placeholder={t("addModel.forecastClassifierConfig.judge.placeholder")}
              onValueChange={(model) => {
                if (model === llm.model) return;
                onChange({
                  ...value,
                  classifier_llm_config: { ...llm, model: model ?? "", reasoning_effort: undefined },
                });
              }}
            />
          </div>
        </>
      )}
      {section !== "advanced" && (
        <>
          {isCapability ? (
            <>
              <NumberField
                label={t("addModel.forecastClassifierConfig.solveThreshold.label")}
                value={capability.base_threshold}
                min={0}
                max={1}
                help={t("addModel.forecastClassifierConfig.solveThreshold.help")}
                onChange={(base_threshold) => updateCapability({ ...capability, base_threshold })}
              />
            </>
          ) : (
            <>
              <FuseProfilePresets value={fuse} onChange={updateFuse} />
              <NumberField
                label={t("addModel.forecastClassifierConfig.maxQualityGap.label")}
                value={fuse.max_quality_gap}
                min={0}
                max={1}
                help={t("addModel.forecastClassifierConfig.maxQualityGap.help")}
                onChange={(max_quality_gap) => updateFuse({ ...fuse, max_quality_gap })}
              />
            </>
          )}
        </>
      )}
      {section !== "required" && (
        <Collapsible
          open={section === "advanced" ? true : undefined}
          className={section === "advanced" ? undefined : "rounded-lg border"}
        >
          {section === "all" && (
            <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left font-medium">
              <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" />
              {t("addModel.forecastClassifierConfig.options.title")}
            </CollapsibleTrigger>
          )}
          <CollapsibleContent className="space-y-4 px-4 pb-4">
            <ClassifierReasoningEffortSelect
              model={llm.model}
              value={llm.reasoning_effort}
              explicitlySupported={effortOptionsByModel[llm.model]}
              onChange={(reasoning_effort) =>
                onChange({ ...value, classifier_llm_config: { ...llm, reasoning_effort } })
              }
            />
            <NumberField
              label={t("addModel.forecastClassifierConfig.options.timeout")}
              min={1}
              step={1}
              value={llm.timeout_ms}
              help={t("addModel.forecastClassifierConfig.options.timeoutHelp")}
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
            {isCapability && (
              <NumberField
                label={t("addModel.forecastClassifierConfig.options.boundaryStep")}
                value={capability.threshold_step ?? 0}
                min={0}
                max={0.5}
                help={t("addModel.forecastClassifierConfig.options.boundaryStepHelp")}
                onChange={(threshold_step) => updateCapability({ ...capability, threshold_step })}
              />
            )}
            <NumberField
              label={t("addModel.forecastClassifierConfig.options.outputTokenLimit")}
              min={1}
              step={1}
              value={config.max_output_tokens ?? (isCapability ? 4096 : 1024)}
              onChange={(max_output_tokens) => updateTransport({ max_output_tokens })}
            />
            <div className="space-y-1">
              <Label htmlFor={`${id}-format`}>{t("addModel.forecastClassifierConfig.options.responseFormat")}</Label>
              <SearchSelect
                inputId={`${id}-format`}
                aria-label={t("addModel.forecastClassifierConfig.options.responseFormat")}
                value={config.response_format ?? "json_schema"}
                allowClear={false}
                options={[
                  { value: "json_schema", label: t("addModel.forecastClassifierConfig.options.strictJsonSchema") },
                  { value: "json_object", label: t("addModel.forecastClassifierConfig.options.jsonObject") },
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
                {t("addModel.forecastClassifierConfig.calibration.toggle")}
              </Label>
              <p className="text-xs text-muted-foreground">{t("addModel.forecastClassifierConfig.calibration.help")}</p>
              {config.calibration && (
                <div className="space-y-1">
                  <Label htmlFor={`${id}-version`}>{t("addModel.forecastClassifierConfig.calibration.version")}</Label>
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
                  label={t("addModel.forecastClassifierConfig.calibration.efficient")}
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
                    label={
                      role === "efficient"
                        ? t("addModel.forecastClassifierConfig.calibration.efficient")
                        : t("addModel.forecastClassifierConfig.calibration.capable")
                    }
                    value={fuse.calibration![role]}
                    onChange={(next) => {
                      if (fuse.calibration) updateFuse({ ...fuse, calibration: { ...fuse.calibration, [role]: next } });
                    }}
                  />
                ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
      {section === "all" && (
        <>
          <div className="space-y-1">
            <Label htmlFor={`${id}-frequency`}>{t("addModel.forecastClassifierConfig.frequency.label")}</Label>
            <SearchSelect
              inputId={`${id}-frequency`}
              aria-label={t("addModel.forecastClassifierConfig.frequency.label")}
              value={classificationFrequency(value)}
              allowClear={false}
              options={[
                { value: "every_request", label: t("addModel.forecastClassifierConfig.frequency.everyRequest") },
                { value: "user_turn", label: t("addModel.forecastClassifierConfig.frequency.userTurn") },
                { value: "session", label: t("addModel.forecastClassifierConfig.frequency.session") },
              ]}
              onValueChange={(frequency) => {
                if (frequency) onChange(withClassificationFrequency(value, frequency as ClassificationFrequency));
              }}
            />
          </div>
        </>
      )}
      <p className="text-xs text-muted-foreground">{t("addModel.forecastClassifierConfig.bundledPromptNote")}</p>
      {section !== "advanced" && error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
};

export default ForecastClassifierConfig;
