"use client";

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
  escalated?: boolean;
  tier_boundaries?: RoutingDecisionTierBoundaries;
  reasoning_override_min_score?: number;
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

  const named = (range: string, tier: string): string => (renamed ? range : `${range}, ${tier}`);
  if (score < simpleMedium) return named(`below ${simpleMedium}`, "SIMPLE");
  if (score < mediumComplex) return named(`${simpleMedium} to ${mediumComplex}`, "MEDIUM");
  if (score < complexReasoning) return named(`${mediumComplex} to ${complexReasoning}`, "COMPLEX");
  return named(`at or above ${complexReasoning}`, "REASONING");
}

function describePlanModeFloor(matchedKeyword: string | undefined): string {
  if (matchedKeyword === "exit_plan_mode") return "Plan-mode floor (exit_plan_mode tool)";
  if (matchedKeyword) return `Plan-mode floor: "${matchedKeyword}"`;
  return "Plan-mode floor";
}

/** Rows logged before the floor was recorded name what it tracked back then instead of a number. */
function describeReasoningOverride(tierLabel: string | undefined, floor: number | undefined): string {
  const stated = floor === undefined ? "the Simple to Medium boundary" : String(floor);
  return `Heuristic, ${tierLabel ?? "REASONING"} override (2 or more reasoning markers, score of at least ${stated})`;
}

const CONSTANT_CAUSE_LABELS: Record<string, string> = {
  heuristic_scorer: "Heuristic scorer",
  heuristic_first_short_circuit: "Heuristic scorer, classifier skipped",
  classifier_plugin: "Custom classifier plugin",
  semantic_keyword_match: "Semantic keyword match",
  session_affinity_pin: "Pinned to session",
  session_affinity_escalation: "Escalated from session pin",
  quality_tier: "Quality tier mapping",
  bandit: "Adaptive bandit",
  default_fallback: "Default model, no route matched",
  classifier_fallback: "Fallback tier, LLM classifier failed",
  default_model_fallback: "Default model, LLM classifier failed",
};

function describeCause(decision: RoutingDecision): string {
  const {
    cause,
    classifier_model: classifierModel,
    matched_keyword: matchedKeyword,
    tier_label: tierLabel,
    reasoning_override_min_score: overrideFloor,
  } = decision;

  const constant = cause ? CONSTANT_CAUSE_LABELS[cause] : undefined;
  if (constant) return constant;

  switch (cause) {
    case "reasoning_override":
      return describeReasoningOverride(tierLabel, overrideFloor);
    case "llm_classifier":
      return classifierModel ? `LLM classifier (${classifierModel})` : "LLM classifier";
    case "literal_keyword_match":
    case "keyword":
      return matchedKeyword ? `Keyword match: "${matchedKeyword}"` : "Keyword match";
    case "plan_mode":
      return describePlanModeFloor(matchedKeyword);
    default:
      return cause ?? "Unknown";
  }
}

/**
 * A request can ask to escalate and get nowhere, when its tier is already the highest
 * one configured. That row still has to say the caller asked, otherwise it reads as an
 * ordinary route; it just must not claim a bump that did not happen. Only called when
 * the request escalated or asked to, so there is no "did not escalate" case.
 */
function describeEscalation(escalated: boolean, keyword: string | undefined): string {
  if (escalated) return keyword ? `Yes, keyword "${keyword}"` : "Yes";
  return keyword ? `Requested via "${keyword}"; already at the highest tier` : "Requested; already at the highest tier";
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-sm">
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words">{children}</span>
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
  } = decision;

  // On an override row the score did not decide the tier, so showing it against a
  // boundary would claim something untrue. Keyed off the cause rather than a marker
  // inside `signals`, which redaction can remove.
  const scoreExplanation =
    score !== undefined && decision.cause !== "reasoning_override" && decision.cause !== "plan_mode"
      ? describeScoreAgainstBoundaries(score, tierBoundaries, tierLabel !== undefined)
      : null;

  return (
    <div className={cn("mb-6 w-full max-w-full overflow-hidden rounded-lg bg-card shadow-sm", className)}>
      <div className="border-b px-4 py-2.5 text-sm font-medium">Routing</div>
      <div className="px-4 py-3">
        {routerModelName && (
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Waypoints size={14} aria-hidden />
            <span>{routerModelName}</span>
            {routerType && (
              <span className="font-normal text-muted-foreground">
                ({ROUTER_TYPE_LABELS[routerType] ?? routerType})
              </span>
            )}
          </div>
        )}

        {tier && (
          <Row label="Tier">
            <Badge variant="secondary" className="font-normal">
              {tierLabel ?? tier}
            </Badge>
          </Row>
        )}

        {requestType && <Row label="Request type">{requestType}</Row>}

        <Row label="Decided by">{describeCause(decision)}</Row>

        {score !== undefined && (
          <Row label="Score">
            <span className="tabular-nums">{score.toFixed(2)}</span>
            {scoreExplanation && <span className="ml-2 text-muted-foreground">({scoreExplanation})</span>}
          </Row>
        )}

        {routedModel && <Row label="Routed to">{routedModel}</Row>}

        {escalated !== undefined && <Row label="Escalated">{describeEscalation(escalated, escalationKeyword)}</Row>}

        {signals && signals.length > 0 && (
          <Row label="Signals">
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
