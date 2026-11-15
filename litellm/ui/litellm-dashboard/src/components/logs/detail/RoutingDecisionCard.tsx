"use client";

import type { ParseKeys, TFunction } from "i18next";
import { Waypoints } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import i18n from "@/i18n/i18n";
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

const ROUTER_TYPE_LABELS: Record<string, ParseKeys> = {
  complexity: "logs.detail.routing.routerType.complexity",
  adaptive: "logs.detail.routing.routerType.adaptive",
  quality: "logs.detail.routing.routerType.quality",
};

/**
 * The tier the score alone would have produced, given the boundaries in effect when
 * the decision was made. Rendered as the bracket that explains a score, so it must
 * use the snapshot rather than today's config.
 */
function describeScoreAgainstBoundaries(
  score: number,
  boundaries?: RoutingDecisionTierBoundaries,
  renamed?: boolean,
  t: TFunction = i18n.t.bind(i18n),
): string | null {
  if (!boundaries) return null;
  const {
    simple_medium: simpleMedium,
    medium_complex: mediumComplex,
    complex_reasoning: complexReasoning,
  } = boundaries;
  if (simpleMedium === undefined || mediumComplex === undefined || complexReasoning === undefined) return null;

  const named = (range: string, tier: string): string => (renamed ? range : `${range}, ${tier}`);
  if (score < simpleMedium) return named(t("logs.detail.routing.scoreRange.below", { value: simpleMedium }), "SIMPLE");
  if (score < mediumComplex)
    return named(t("logs.detail.routing.scoreRange.between", { from: simpleMedium, to: mediumComplex }), "MEDIUM");
  if (score < complexReasoning)
    return named(t("logs.detail.routing.scoreRange.between", { from: mediumComplex, to: complexReasoning }), "COMPLEX");
  return named(t("logs.detail.routing.scoreRange.atOrAbove", { value: complexReasoning }), "REASONING");
}

function describePlanModeFloor(matchedKeyword: string | undefined, t: TFunction): string {
  if (matchedKeyword === "exit_plan_mode") return t("logs.detail.routing.cause.planModeExitTool");
  if (matchedKeyword) return t("logs.detail.routing.cause.planModeKeyword", { keyword: matchedKeyword });
  return t("logs.detail.routing.cause.planMode");
}

/**
 * The sentinel is the whole reason this row is worth reading: it is the string an operator
 * would add to housekeeping_patterns to cover another client, so naming it turns the row into
 * the instruction. Without it the drawer says only that the classifier was skipped.
 */
function describeHousekeeping(matchedKeyword: string | undefined, t: TFunction): string {
  if (matchedKeyword) return t("logs.detail.routing.cause.housekeepingKeyword", { keyword: matchedKeyword });
  return t("logs.detail.routing.cause.housekeeping");
}

/** Rows logged before the floor was recorded name what it tracked back then instead of a number. */
function describeReasoningOverride(tierLabel: string | undefined, floor: number | undefined, t: TFunction): string {
  const stated = floor === undefined ? t("logs.detail.routing.cause.simpleMediumBoundary") : String(floor);
  return t("logs.detail.routing.cause.reasoningOverride", { tier: tierLabel ?? "REASONING", floor: stated });
}

const CONSTANT_CAUSE_LABELS: Record<string, ParseKeys> = {
  heuristic_scorer: "logs.detail.routing.cause.heuristicScorer",
  heuristic_v2: "logs.detail.routing.cause.heuristicV2",
  capability_classifier: "logs.detail.routing.cause.capabilityClassifier",
  capability_classifier_fallback: "logs.detail.routing.cause.capabilityClassifierFallback",
  llm_v2_classifier: "logs.detail.routing.cause.llmV2Classifier",
  llm_v2_fallback: "logs.detail.routing.cause.llmV2Fallback",
  heuristic_first_short_circuit: "logs.detail.routing.cause.heuristicFirstShortCircuit",
  hybrid_short_circuit: "logs.detail.routing.cause.hybridShortCircuit",
  classifier_plugin: "logs.detail.routing.cause.classifierPlugin",
  semantic_keyword_match: "logs.detail.routing.cause.semanticKeywordMatch",
  session_affinity_pin: "logs.detail.routing.cause.sessionAffinityPin",
  session_affinity_escalation: "logs.detail.routing.cause.sessionAffinityEscalation",
  user_turn_continuation: "logs.detail.routing.cause.userTurnContinuation",
  modality_escalation: "logs.detail.routing.cause.modalityEscalation",
  modality_pin_override: "logs.detail.routing.cause.modalityPinOverride",
  quality_tier: "logs.detail.routing.cause.qualityTier",
  bandit: "logs.detail.routing.cause.bandit",
  default_fallback: "logs.detail.routing.cause.defaultFallback",
  classifier_fallback: "logs.detail.routing.cause.classifierFallback",
  default_model_fallback: "logs.detail.routing.cause.defaultModelFallback",
};

function describeCause(decision: RoutingDecision, t: TFunction): string {
  const {
    cause,
    classifier_model: classifierModel,
    matched_keyword: matchedKeyword,
    tier_label: tierLabel,
    reasoning_override_min_score: overrideFloor,
  } = decision;

  const constant = cause && Object.hasOwn(CONSTANT_CAUSE_LABELS, cause) ? CONSTANT_CAUSE_LABELS[cause] : undefined;
  if (constant) return t(constant);

  switch (cause) {
    case "reasoning_override":
      return describeReasoningOverride(tierLabel, overrideFloor, t);
    case "llm_classifier":
      return classifierModel
        ? t("logs.detail.routing.cause.llmClassifierModel", { model: classifierModel })
        : t("logs.detail.routing.cause.llmClassifier");
    case "jev_classifier":
      return t("logs.detail.routing.cause.ossClassifier");
    case "literal_keyword_match":
    case "keyword":
      return matchedKeyword
        ? t("logs.detail.routing.cause.keywordMatchKeyword", { keyword: matchedKeyword })
        : t("logs.detail.routing.cause.keywordMatch");
    case "plan_mode":
      return describePlanModeFloor(matchedKeyword, t);
    case "housekeeping":
      return describeHousekeeping(matchedKeyword, t);
    default:
      return cause ?? t("logs.detail.routing.cause.unknown");
  }
}

/**
 * A request can ask to escalate and get nowhere, when its tier is already the highest
 * one configured. That row still has to say the caller asked, otherwise it reads as an
 * ordinary route; it just must not claim a bump that did not happen. Only called when
 * the request escalated or asked to, so there is no "did not escalate" case.
 */
function describeEscalation(escalated: boolean, keyword: string | undefined, t: TFunction): string {
  if (escalated)
    return keyword
      ? t("logs.detail.routing.escalation.yesKeyword", { keyword })
      : t("logs.detail.routing.escalation.yes");
  return keyword
    ? t("logs.detail.routing.escalation.requestedKeyword", { keyword })
    : t("logs.detail.routing.escalation.requested");
}

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
  const { t } = useTranslation();
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
      <div className="mb-1 text-sm font-medium">{t("logs.detail.routing.capability.title")}</div>
      <div className="mb-2 text-xs text-muted-foreground">{t("logs.detail.routing.capability.subtitle")}</div>
      <PercentageRow label={t("logs.detail.routing.capability.raw")} value={raw} />
      <PercentageRow label={t("logs.detail.routing.capability.calibrated")} value={calibrated} />
      <PercentageRow label={t("logs.detail.routing.capability.threshold")} value={threshold} />
      {boundary && <Row label={t("logs.detail.routing.capability.boundary")}>{boundary}</Row>}
      {rule && <Row label={t("logs.detail.routing.capability.rule")}>{rule}</Row>}
      {version && <Row label={t("logs.detail.routing.calibration")}>{version}</Row>}
    </div>
  );
}

function FuseV2Forecast({ decision }: { decision: RoutingDecision }) {
  const { t } = useTranslation();
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
      <div className="mb-1 text-sm font-medium">{t("logs.detail.routing.fuseV2.title")}</div>
      <PercentageRow label={t("logs.detail.routing.fuseV2.efficientRaw")} value={rawEfficient} />
      <PercentageRow label={t("logs.detail.routing.fuseV2.capableRaw")} value={rawCapable} />
      <PercentageRow label={t("logs.detail.routing.fuseV2.efficientCalibrated")} value={calibratedEfficient} />
      <PercentageRow label={t("logs.detail.routing.fuseV2.capableCalibrated")} value={calibratedCapable} />
      <PercentageRow
        label={t("logs.detail.routing.fuseV2.appliedGap")}
        value={gap}
        unit={t("logs.detail.routing.fuseV2.percentagePoints")}
      />
      <PercentageRow
        label={t("logs.detail.routing.fuseV2.allowedGap")}
        value={allowedGap}
        unit={t("logs.detail.routing.fuseV2.percentagePoints")}
      />
      {version && <Row label={t("logs.detail.routing.calibration")}>{version}</Row>}
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
  const { t } = useTranslation();
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
      ? describeScoreAgainstBoundaries(score, tierBoundaries, tierLabel !== undefined, t)
      : null;

  return (
    <div className={cn("mb-6 w-full max-w-full overflow-hidden rounded-lg bg-card shadow-sm", className)}>
      <div className="border-b px-4 py-2.5 text-sm font-medium">{t("logs.detail.routing.title")}</div>
      <div className="px-4 py-3">
        {routerModelName && (
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Waypoints size={14} aria-hidden />
            <span>{routerModelName}</span>
            {routerType && (
              <span className="font-normal text-muted-foreground">
                ({Object.hasOwn(ROUTER_TYPE_LABELS, routerType) ? t(ROUTER_TYPE_LABELS[routerType]) : routerType})
              </span>
            )}
          </div>
        )}

        {tier && (
          <Row label={t("logs.detail.routing.tier")}>
            <Badge variant="secondary" className="font-normal">
              {tierLabel ?? tier}
            </Badge>
          </Row>
        )}

        {requestType && <Row label={t("logs.detail.routing.requestType")}>{requestType}</Row>}

        <Row label={t("logs.detail.routing.decidedBy")}>{describeCause(decision, t)}</Row>
        {decision.classifier_model && (
          <Row label={t("logs.detail.routing.classifierModel")}>{decision.classifier_model}</Row>
        )}
        {decision.classifier_confidence != null && (
          <Row label={t("logs.detail.routing.confidence")}>{(decision.classifier_confidence * 100).toFixed(1)}%</Row>
        )}
        {decision.classifier_probabilities && (
          <Row label={t("logs.detail.routing.probabilities")}>
            {Object.entries(decision.classifier_probabilities).map(([name, probability]) => (
              <div key={name}>
                {name}: {(probability * 100).toFixed(1)}%
              </div>
            ))}
          </Row>
        )}
        {decision.classifier_cost != null && (
          <Row label={t("logs.detail.routing.classifierCost")}>${decision.classifier_cost.toFixed(8)}</Row>
        )}

        {score !== undefined && (
          <Row label={t("logs.detail.routing.score")}>
            <span className="tabular-nums">{score.toFixed(2)}</span>
            {scoreExplanation && <span className="ml-2 text-muted-foreground">({scoreExplanation})</span>}
          </Row>
        )}

        {routedModel && <Row label={t("logs.detail.routing.routedTo")}>{routedModel}</Row>}

        {escalated !== undefined && (
          <Row label={t("logs.detail.routing.escalated")}>{describeEscalation(escalated, escalationKeyword, t)}</Row>
        )}

        {forecast && (
          <div className="mt-3 border-t pt-3">
            <div className="mb-1 text-sm font-medium">{t("logs.detail.routing.heuristicV2.title")}</div>
            <Row label={t("logs.detail.routing.heuristicV2.successByTier")}>
              <span className="flex flex-wrap gap-1">
                {["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"].map((predictedTier) => (
                  <Badge key={predictedTier} variant="outline" className="font-normal tabular-nums">
                    {predictedTier} {(forecast.probabilities[predictedTier] * 100).toFixed(1)}%
                  </Badge>
                ))}
              </span>
            </Row>
            <Row label={t("logs.detail.routing.heuristicV2.threshold")}>
              <span className="tabular-nums">{(forecast.threshold * 100).toFixed(1)}%</span>
            </Row>
            <Row label={t("logs.detail.routing.heuristicV2.predictedTier")}>{forecast.predicted_tier}</Row>
            <Row label={t("logs.detail.routing.requestType")}>{forecast.request_type}</Row>
          </div>
        )}

        <CapabilityForecast decision={decision} />
        <FuseV2Forecast decision={decision} />

        {signals && signals.length > 0 && (
          <Row label={t("logs.detail.routing.signals")}>
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
