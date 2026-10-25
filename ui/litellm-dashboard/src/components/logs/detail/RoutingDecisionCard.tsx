"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import type { Translate } from "@/i18n/language";
import { Waypoints } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

export interface RoutingDecisionTierBoundaries {
  simple_medium?: number;
  medium_complex?: number;
  complex_reasoning?: number;
}

export interface RoutingDecision {
  router_model_name?: string;
  router_type?: string;
  routed_model?: string;
  cause?: string;
  tier?: string;
  tier_label?: string;
  request_type?: string;
  score?: number;
  signals?: string[];
  matched_keyword?: string;
  escalation_keyword?: string;
  classifier_model?: string;
  classifier_p_solve?: number;
  classifier_calibrated_p_solve?: number;
  classifier_threshold?: number;
  classifier_capability_boundary?: string;
  classifier_primary_rule?: string;
  classifier_calibration_version?: string;
  classifier_efficient_p_solve?: number;
  classifier_capable_p_solve?: number;
  classifier_calibrated_efficient_p_solve?: number;
  classifier_calibrated_capable_p_solve?: number;
  classifier_max_quality_gap?: number;
  classifier_confidence?: number;
  classifier_probabilities?: Record<string, number>;
  classifier_cost?: number;
  escalated?: boolean;
  tier_boundaries?: RoutingDecisionTierBoundaries;
  reasoning_override_min_score?: number;
  heuristic_v2_forecast?: {
    probabilities: Record<string, number>;
    threshold: number;
    predicted_tier: string;
    request_type: string;
  };
}

const ROUTER_TYPE_LABELS: Record<string, string> = {
  complexity: "Auto-Router v2",
  adaptive: "Adaptive router",
  quality: "Quality router",
};

/**
 * The tier the score alone would have produced, given the boundaries in effect when
 * the decision was made. Rendered as the bracket that explains a score, so it must
 * use the snapshot rather than today's config.
 */
function describeScoreAgainstBoundaries(
  score: number,
  t: Translate,
  boundaries?: RoutingDecisionTierBoundaries,
  renamed?: boolean,
): string | null {
  if (!boundaries) return null;
  const {
    simple_medium: simpleMedium,
    medium_complex: mediumComplex,
    complex_reasoning: complexReasoning,
  } = boundaries;
  if (simpleMedium === undefined || mediumComplex === undefined || complexReasoning === undefined) return null;

  const named = (range: string, tier: string): string =>
    renamed ? range : t("{range}, {tier}", { range, tier: t(tier) });
  if (score < simpleMedium) return named(t("below {boundary}", { boundary: simpleMedium }), "SIMPLE");
  if (score < mediumComplex)
    return named(t("{lower} to {upper}", { lower: simpleMedium, upper: mediumComplex }), "MEDIUM");
  if (score < complexReasoning)
    return named(t("{lower} to {upper}", { lower: mediumComplex, upper: complexReasoning }), "COMPLEX");
  return named(t("at or above {boundary}", { boundary: complexReasoning }), "REASONING");
}

function describePlanModeFloor(matchedKeyword: string | undefined, t: Translate): string {
  if (matchedKeyword === "exit_plan_mode") return t("Plan-mode floor (exit_plan_mode tool)");
  if (matchedKeyword) return t('Plan-mode floor: "{keyword}"', { keyword: matchedKeyword });
  return t("Plan-mode floor");
}

/**
 * The sentinel is the whole reason this row is worth reading: it is the string an operator
 * would add to housekeeping_patterns to cover another client, so naming it turns the row into
 * the instruction. Without it the drawer says only that the classifier was skipped.
 */
function describeHousekeeping(matchedKeyword: string | undefined, t: Translate): string {
  if (matchedKeyword) return t('Client housekeeping call: "{keyword}"', { keyword: matchedKeyword });
  return t("Client housekeeping call, classifier skipped");
}

/** Rows logged before the floor was recorded name what it tracked back then instead of a number. */
function describeReasoningOverride(tierLabel: string | undefined, floor: number | undefined, t: Translate): string {
  const stated = floor === undefined ? t("the Simple to Medium boundary") : String(floor);
  return t("Heuristic, {tier} override (2 or more reasoning markers, score of at least {floor})", {
    tier: tierLabel ?? t("REASONING"),
    floor: stated,
  });
}

const CONSTANT_CAUSE_LABELS: Record<string, string> = {
  heuristic_scorer: "Heuristic scorer",
  heuristic_v2: "Heuristic v2",
  capability_classifier: "Capability",
  capability_classifier_fallback: "Capable tier, Capability classifier failed",
  llm_v2_classifier: "FUSE v2",
  llm_v2_fallback: "Capable tier, FUSE v2 classifier failed",
  heuristic_first_short_circuit: "Heuristic scorer, classifier skipped",
  hybrid_short_circuit: "Heuristic scorer, score clear of every boundary",
  classifier_plugin: "Custom classifier plugin",
  semantic_keyword_match: "Semantic keyword match",
  session_affinity_pin: "Pinned to session",
  session_affinity_escalation: "Escalated from session pin",
  user_turn_continuation: "Continuation turn, classifier skipped",
  modality_escalation: "Escalated for image input",
  modality_pin_override: "Overrode session pin for image input",
  quality_tier: "Quality tier mapping",
  bandit: "Adaptive bandit",
  default_fallback: "Default model, no route matched",
  classifier_fallback: "Fallback tier, classifier failed",
  default_model_fallback: "Default model, classifier failed",
};

function describeCause(decision: RoutingDecision, t: Translate): string {
  const {
    cause,
    classifier_model: classifierModel,
    matched_keyword: matchedKeyword,
    tier_label: tierLabel,
    reasoning_override_min_score: overrideFloor,
  } = decision;

  const constant = cause ? CONSTANT_CAUSE_LABELS[cause] : undefined;
  if (constant) return t(constant);

  switch (cause) {
    case "reasoning_override":
      return describeReasoningOverride(tierLabel, overrideFloor, t);
    case "llm_classifier":
      return classifierModel ? t("LLM classifier ({model})", { model: classifierModel }) : t("LLM classifier");
    case "jev_classifier":
      return t("OSS classifier");
    case "literal_keyword_match":
    case "keyword":
      return matchedKeyword ? t('Keyword match: "{keyword}"', { keyword: matchedKeyword }) : t("Keyword match");
    case "plan_mode":
      return describePlanModeFloor(matchedKeyword, t);
    case "housekeeping":
      return describeHousekeeping(matchedKeyword, t);
    default:
      return cause ?? t("Unknown");
  }
}

/**
 * A request can ask to escalate and get nowhere, when its tier is already the highest
 * one configured. That row still has to say the caller asked, otherwise it reads as an
 * ordinary route; it just must not claim a bump that did not happen. Only called when
 * the request escalated or asked to, so there is no "did not escalate" case.
 */
function describeEscalation(escalated: boolean, keyword: string | undefined, t: Translate): string {
  if (escalated) return keyword ? t('Yes, keyword "{keyword}"', { keyword }) : t("Yes");
  return keyword
    ? t('Requested via "{keyword}"; already at the highest tier', { keyword })
    : t("Requested; already at the highest tier");
}

// 内建层级使用中文标签，自定义层级和后端自定义名称不参与词典匹配。
const displayTier = (tier: string, t: Translate): string =>
  ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING", "efficient", "capable"].includes(tier) ? t(tier) : tier;

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-sm">
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}

function PercentageRow({ label, value, unit = "%" }: { label: string; value?: number; unit?: string }) {
  if (value === undefined) return null;
  return (
    <Row label={label}>
      <span className="tabular-nums">{`${(value * 100).toFixed(1)}${unit}`}</span>
    </Row>
  );
}

function CapabilityForecast({ decision }: { decision: RoutingDecision }) {
  const { t } = useLanguage();
  const {
    classifier_p_solve: raw,
    classifier_calibrated_p_solve: calibrated,
    classifier_threshold: threshold,
    classifier_capability_boundary: boundary,
    classifier_primary_rule: rule,
    classifier_calibration_version: version,
  } = decision;
  if ([raw, calibrated, threshold, boundary, rule].every((value) => value === undefined)) return null;

  return (
    <div className="mt-3 border-t pt-3">
      <div className="mb-1 text-sm font-medium">{t("Capability estimates")}</div>
      <div className="mb-2 text-xs text-muted-foreground">{t("Efficient model solve chance")}</div>
      <PercentageRow label={t("Raw")} value={raw} />
      <PercentageRow label={t("Calibrated")} value={calibrated} />
      <PercentageRow label={t("Threshold")} value={threshold} />
      {boundary && <Row label={t("Boundary")}>{boundary}</Row>}
      {rule && <Row label={t("Rule")}>{rule}</Row>}
      {version && <Row label={t("Calibration")}>{version}</Row>}
    </div>
  );
}

function FuseV2Forecast({ decision }: { decision: RoutingDecision }) {
  const { t } = useLanguage();
  const {
    classifier_efficient_p_solve: rawEfficient,
    classifier_capable_p_solve: rawCapable,
    classifier_calibrated_efficient_p_solve: calibratedEfficient,
    classifier_calibrated_capable_p_solve: calibratedCapable,
    classifier_max_quality_gap: allowedGap,
    classifier_calibration_version: version,
  } = decision;
  if ([rawEfficient, rawCapable, calibratedEfficient, calibratedCapable, allowedGap].every((v) => v === undefined)) {
    return null;
  }
  const isCalibrated = [calibratedEfficient, calibratedCapable, version].some((value) => value !== undefined);
  const efficient = isCalibrated ? calibratedEfficient : rawEfficient;
  const capable = isCalibrated ? calibratedCapable : rawCapable;
  const gap = efficient !== undefined && capable !== undefined ? capable - efficient : undefined;

  return (
    <div className="mt-3 border-t pt-3">
      <div className="mb-1 text-sm font-medium">{t("FUSE v2 estimates")}</div>
      <PercentageRow label={t("Efficient (raw)")} value={rawEfficient} />
      <PercentageRow label={t("Capable (raw)")} value={rawCapable} />
      <PercentageRow label={t("Efficient (calibrated)")} value={calibratedEfficient} />
      <PercentageRow label={t("Capable (calibrated)")} value={calibratedCapable} />
      <PercentageRow label={t("Applied gap")} value={gap} unit={t(" percentage points")} />
      <PercentageRow label={t("Allowed gap")} value={allowedGap} unit={t(" percentage points")} />
      {version && <Row label={t("Calibration")}>{version}</Row>}
    </div>
  );
}

export function RoutingDecisionCard({
  decision,
  className,
}: {
  decision?: RoutingDecision | null;
  className?: string;
}) {
  const { t } = useLanguage();
  if (!decision || !decision.cause) return null;

  const {
    router_model_name: routerModelName,
    router_type: routerType,
    routed_model: routedModel,
    tier,
    tier_label: tierLabel,
    request_type: requestType,
    score,
    signals,
    escalated,
    escalation_keyword: escalationKeyword,
    tier_boundaries: tierBoundaries,
    heuristic_v2_forecast: forecast,
  } = decision;

  // On an override row the score did not decide the tier, so showing it against a
  // boundary would claim something untrue. Keyed off the cause rather than a marker
  // inside `signals`, which redaction can remove.
  const scoreExplanation =
    score !== undefined && decision.cause !== "reasoning_override" && decision.cause !== "plan_mode"
      ? describeScoreAgainstBoundaries(score, t, tierBoundaries, tierLabel !== undefined)
      : null;

  return (
    <div className={cn("mb-6 w-full max-w-full overflow-hidden rounded-lg bg-card shadow-sm", className)}>
      <div className="border-b px-4 py-2.5 text-sm font-medium">{t("Routing")}</div>
      <div className="px-4 py-3">
        {routerModelName && (
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Waypoints size={14} aria-hidden />
            <span>{routerModelName}</span>
            {routerType && (
              <span className="font-normal text-muted-foreground">
                ({ROUTER_TYPE_LABELS[routerType] ? t(ROUTER_TYPE_LABELS[routerType]) : routerType})
              </span>
            )}
          </div>
        )}

        {tier && (
          <Row label={t("Tier")}>
            <Badge variant="secondary" className="font-normal">
              {tierLabel ?? displayTier(tier, t)}
            </Badge>
          </Row>
        )}

        {requestType && <Row label={t("Request type")}>{requestType}</Row>}

        <Row label={t("Decided by")}>{describeCause(decision, t)}</Row>
        {decision.classifier_model && <Row label={t("Classifier model")}>{decision.classifier_model}</Row>}
        {decision.classifier_confidence != null && (
          <Row label={t("Confidence")}>{(decision.classifier_confidence * 100).toFixed(1)}%</Row>
        )}
        {decision.classifier_probabilities && (
          <Row label={t("Probabilities")}>
            {Object.entries(decision.classifier_probabilities).map(([name, probability]) => (
              <div key={name}>
                {displayTier(name, t)}: {(probability * 100).toFixed(1)}%
              </div>
            ))}
          </Row>
        )}
        {decision.classifier_cost != null && (
          <Row label={t("Classifier cost")}>${decision.classifier_cost.toFixed(8)}</Row>
        )}

        {score !== undefined && (
          <Row label={t("Score")}>
            <span className="tabular-nums">{score.toFixed(2)}</span>
            {scoreExplanation && <span className="ml-2 text-muted-foreground">({scoreExplanation})</span>}
          </Row>
        )}

        {routedModel && <Row label={t("Routed to")}>{routedModel}</Row>}

        {escalated !== undefined && (
          <Row label={t("Escalated")}>{describeEscalation(escalated, escalationKeyword, t)}</Row>
        )}

        {forecast && (
          <div className="mt-3 border-t pt-3">
            <div className="mb-1 text-sm font-medium">{t("Heuristic v2 estimates")}</div>
            <Row label={t("Success by tier")}>
              <span className="flex flex-wrap gap-1">
                {["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"].map((predictedTier) => (
                  <Badge key={predictedTier} variant="outline" className="font-normal tabular-nums">
                    {displayTier(predictedTier, t)} {(forecast.probabilities[predictedTier] * 100).toFixed(1)}%
                  </Badge>
                ))}
              </span>
            </Row>
            <Row label={t("Threshold")}>
              <span className="tabular-nums">{(forecast.threshold * 100).toFixed(1)}%</span>
            </Row>
            <Row label={t("Predicted tier")}>{displayTier(forecast.predicted_tier, t)}</Row>
            <Row label={t("Request type")}>{forecast.request_type}</Row>
          </div>
        )}

        <CapabilityForecast decision={decision} />
        <FuseV2Forecast decision={decision} />

        {signals && signals.length > 0 && (
          <Row label={t("Signals")}>
            <span className="flex flex-wrap gap-1">
              {signals.map((signal) => (
                <Badge key={signal} variant="outline" className="font-normal">
                  {signal}
                </Badge>
              ))}
            </span>
          </Row>
        )}
      </div>
    </div>
  );
}

export default RoutingDecisionCard;
