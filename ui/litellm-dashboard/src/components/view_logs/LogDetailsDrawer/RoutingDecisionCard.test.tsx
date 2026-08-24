import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RoutingDecisionCard, type RoutingDecision } from "./RoutingDecisionCard";

const heuristic: RoutingDecision = {
  router_model_name: "smart-router",
  router_type: "complexity",
  routed_model: "claude-sonnet",
  cause: "heuristic_scorer",
  tier: "REASONING",
  score: 0.82,
  signals: ["long (900 tokens)", "code (python, function)"],
  tier_boundaries: { simple_medium: 0.15, medium_complex: 0.35, complex_reasoning: 0.6 },
};

describe("RoutingDecisionCard", () => {
  it("renders nothing when the request carried no routing decision", () => {
    const { container } = render(<RoutingDecisionCard decision={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("explains a heuristic score against the boundaries that were in effect", () => {
    render(<RoutingDecisionCard decision={heuristic} />);
    expect(screen.getByText("smart-router")).toBeInTheDocument();
    expect(screen.getByText("(Auto-Router v2)")).toBeInTheDocument();
    expect(screen.getByText("REASONING")).toBeInTheDocument();
    expect(screen.getByText("Heuristic scorer")).toBeInTheDocument();
    expect(screen.getByText("0.82")).toBeInTheDocument();
    expect(screen.getByText("(at or above 0.6, REASONING)")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet")).toBeInTheDocument();
    expect(screen.getByText("long (900 tokens)")).toBeInTheDocument();
  });

  it("uses the persisted boundary snapshot, not today's defaults", () => {
    // Same score, boundaries the operator had configured lower: it lands in a
    // different band, and the card must say so.
    render(
      <RoutingDecisionCard
        decision={{
          ...heuristic,
          score: 0.4,
          tier: "REASONING",
          tier_boundaries: { simple_medium: 0.1, medium_complex: 0.2, complex_reasoning: 0.3 },
        }}
      />,
    );
    expect(screen.getByText("(at or above 0.3, REASONING)")).toBeInTheDocument();
  });

  it("labels a reasoning override and does not claim the score met a boundary", () => {
    render(
      <RoutingDecisionCard
        decision={{
          ...heuristic,
          cause: "reasoning_override",
          score: 0.2,
          signals: ["reasoning (prove, step-by-step)"],
        }}
      />,
    );
    expect(
      screen.getByText(
        "Heuristic, REASONING override (2 or more reasoning markers, score of at least the Simple to Medium boundary)",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("0.20")).toBeInTheDocument();
    // The score did not decide this tier, so NO band explanation may render at all.
    // Asserting the absence of one specific band would pass vacuously: 0.20 sits in
    // the MEDIUM band, so the REASONING wording is absent either way.
    expect(screen.queryByText(/SIMPLE|MEDIUM|COMPLEX|at or above/)).not.toBeInTheDocument();
  });

  it("names the judge model on the LLM classifier path and shows no score", () => {
    render(
      <RoutingDecisionCard
        decision={{
          router_model_name: "llm-router",
          router_type: "complexity",
          routed_model: "claude-sonnet",
          cause: "llm_classifier",
          tier: "REASONING",
          classifier_model: "claude-haiku",
          signals: ["llm-classifier:REASONING"],
        }}
      />,
    );
    expect(screen.getByText("LLM classifier (claude-haiku)")).toBeInTheDocument();
    expect(screen.queryByText("Score")).not.toBeInTheDocument();
  });

  it("explains a route that fell back to the default model after the classifier failed", () => {
    // No tier is recorded on this path, so the card must not show a Tier row: nothing
    // about the request produced one, the classifier never answered.
    render(
      <RoutingDecisionCard
        decision={{
          router_model_name: "llm-router",
          router_type: "complexity",
          routed_model: "gpt-4o",
          cause: "default_model_fallback",
          signals: ["classifier-failed:default-model"],
        }}
      />,
    );
    expect(screen.getByText("Default model, LLM classifier failed")).toBeInTheDocument();
    expect(screen.queryByText("Tier")).not.toBeInTheDocument();
  });

  it("explains a route that fell back to the configured fallback tier after the classifier failed", () => {
    render(
      <RoutingDecisionCard
        decision={{
          router_model_name: "custom-tier-router",
          router_type: "complexity",
          routed_model: "claude-sonnet",
          cause: "classifier_fallback",
          tier: "SECURITY_REVIEW",
          signals: ["classifier-fallback:SECURITY_REVIEW"],
        }}
      />,
    );
    expect(screen.getByText("Fallback tier, LLM classifier failed")).toBeInTheDocument();
    expect(screen.getByText("SECURITY_REVIEW")).toBeInTheDocument();
  });

  it("shows the keyword that fired a tier rule", () => {
    render(
      <RoutingDecisionCard
        decision={{ ...heuristic, cause: "literal_keyword_match", matched_keyword: "deploy to k8s", score: undefined }}
      />,
    );
    expect(screen.getByText('Keyword match: "deploy to k8s"')).toBeInTheDocument();
  });

  it("shows the plan-mode sentinel that floored the tier", () => {
    render(
      <RoutingDecisionCard
        decision={{ ...heuristic, cause: "plan_mode", matched_keyword: "Plan mode is active", score: undefined }}
      />,
    );
    expect(screen.getByText('Plan-mode floor: "Plan mode is active"')).toBeInTheDocument();
  });

  it("names the exit_plan_mode tool instead of quoting it as a sentinel", () => {
    render(
      <RoutingDecisionCard
        decision={{ ...heuristic, cause: "plan_mode", matched_keyword: "exit_plan_mode", score: undefined }}
      />,
    );
    expect(screen.getByText("Plan-mode floor (exit_plan_mode tool)")).toBeInTheDocument();
  });

  it("does not claim the score chose the tier on a plan-mode floored row", () => {
    // The score's band can name a lower tier than the floored badge; the cause suppresses it.
    render(
      <RoutingDecisionCard decision={{ ...heuristic, cause: "plan_mode", matched_keyword: "Plan mode is active" }} />,
    );
    expect(screen.queryByText(/below|to 0|at or above/)).not.toBeInTheDocument();
    expect(screen.getByText('Plan-mode floor: "Plan mode is active"')).toBeInTheDocument();
  });

  it("shows the escalation keyword", () => {
    render(
      <RoutingDecisionCard decision={{ ...heuristic, escalated: true, escalation_keyword: "LITELLM ESCALATE" }} />,
    );
    expect(screen.getByText('Yes, keyword "LITELLM ESCALATE"')).toBeInTheDocument();
  });

  it("still shows the ask when escalation had nowhere higher to go", () => {
    // The tier did not move, but the row must not read like a request that never
    // asked to escalate.
    render(
      <RoutingDecisionCard decision={{ ...heuristic, escalated: false, escalation_keyword: "LITELLM ESCALATE" }} />,
    );
    expect(screen.getByText('Requested via "LITELLM ESCALATE"; already at the highest tier')).toBeInTheDocument();
  });

  it("omits the escalation row when no escalation was requested", () => {
    render(<RoutingDecisionCard decision={heuristic} />);
    expect(screen.queryByText("Escalated")).not.toBeInTheDocument();
  });

  it("still shows a ceiling escalation after the keyword is redacted away", () => {
    // Under message redaction the keyword is gone but `escalated` survives, so the
    // row must still say an escalation was requested.
    render(<RoutingDecisionCard decision={{ ...heuristic, escalated: false }} />);
    expect(screen.getByText("Requested; already at the highest tier")).toBeInTheDocument();
  });

  it("does not claim the score chose the tier on a redacted override row", () => {
    // `signals` is gone under redaction; the cause alone must suppress the band.
    render(<RoutingDecisionCard decision={{ ...heuristic, cause: "reasoning_override", signals: undefined }} />);
    expect(screen.queryByText(/SIMPLE|MEDIUM|COMPLEX|at or above/)).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "Heuristic, REASONING override (2 or more reasoning markers, score of at least the Simple to Medium boundary)",
      ),
    ).toBeInTheDocument();
  });

  it("shows the operator's tier name on the badge instead of the canonical one", () => {
    render(<RoutingDecisionCard decision={{ ...heuristic, tier_label: "Deep" }} />);
    expect(screen.getByText("Deep")).toBeInTheDocument();
    expect(screen.queryByText("REASONING")).not.toBeInTheDocument();
  });

  it("keeps the canonical tier name when the router did not rename it", () => {
    render(<RoutingDecisionCard decision={heuristic} />);
    expect(screen.getByText("REASONING")).toBeInTheDocument();
  });

  it("drops the tier name from the score band on a renamed router", () => {
    render(<RoutingDecisionCard decision={{ ...heuristic, tier_label: "Deep" }} />);
    expect(screen.getByText("(at or above 0.6)")).toBeInTheDocument();
    expect(screen.queryByText(/at or above 0\.6, REASONING/)).not.toBeInTheDocument();
  });

  it("uses the operator's tier name in the reasoning override description", () => {
    render(
      <RoutingDecisionCard decision={{ ...heuristic, cause: "reasoning_override", score: 0.2, tier_label: "Deep" }} />,
    );
    expect(
      screen.getByText(
        "Heuristic, Deep override (2 or more reasoning markers, score of at least the Simple to Medium boundary)",
      ),
    ).toBeInTheDocument();
  });

  it("states the floor the override actually cleared", () => {
    render(
      <RoutingDecisionCard
        decision={{ ...heuristic, cause: "reasoning_override", score: 0.2, reasoning_override_min_score: 0.05 }}
      />,
    );
    expect(
      screen.getByText("Heuristic, REASONING override (2 or more reasoning markers, score of at least 0.05)"),
    ).toBeInTheDocument();
  });

  // A floor of 0 is an unconditional override, so a falsy check here would print the "before this change"
  // wording on a row that recorded a real floor.
  it("states a recorded floor of 0 rather than treating it as unrecorded", () => {
    render(
      <RoutingDecisionCard
        decision={{ ...heuristic, cause: "reasoning_override", score: 0.2, reasoning_override_min_score: 0 }}
      />,
    );
    expect(
      screen.getByText("Heuristic, REASONING override (2 or more reasoning markers, score of at least 0)"),
    ).toBeInTheDocument();
  });

  it("never prints undefined on a row logged before the floor was recorded", () => {
    render(<RoutingDecisionCard decision={{ ...heuristic, cause: "reasoning_override", score: 0.2 }} />);
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });

  it("falls back to the raw cause for a value this build does not know", () => {
    render(<RoutingDecisionCard decision={{ cause: "some_future_cause", routed_model: "m" }} />);
    expect(screen.getByText("some_future_cause")).toBeInTheDocument();
  });
});
