import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { RoutingDecisionCard, type RoutingDecision } from "./RoutingDecisionCard";

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}));

vi.mock("lucide-react", () => ({ Waypoints: () => null }));

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

const forecast = {
  probabilities: { MEDIUM: 0.69321, SIMPLE: 0, COMPLEX: 0.81234, REASONING: 0.92345 },
  threshold: 0.69,
  predicted_tier: "MEDIUM",
  request_type: "code_generation",
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
    expect(screen.queryByText("Heuristic v2 estimates")).not.toBeInTheDocument();
  });

  it("shows recorded v2 success estimates, including zero, when signals were redacted", () => {
    render(
      <RoutingDecisionCard
        decision={{
          cause: "heuristic_v2",
          tier: "MEDIUM",
          tier_label: "Balanced",
          heuristic_v2_forecast: forecast,
        }}
      />,
    );

    expect(screen.getByText("Heuristic v2 estimates")).toBeInTheDocument();
    expect(screen.getByText("Success by tier")).toBeInTheDocument();
    expect(
      screen.getAllByText(/^(SIMPLE|MEDIUM|COMPLEX|REASONING) \d+\.\d%$/).map((badge) => badge.textContent),
    ).toEqual(["SIMPLE 0.0%", "MEDIUM 69.3%", "COMPLEX 81.2%", "REASONING 92.3%"]);
    expect(screen.getByText("Threshold")).toBeInTheDocument();
    expect(screen.getByText("69.0%")).toBeInTheDocument();
    expect(screen.getByText("Predicted tier")).toBeInTheDocument();
    expect(screen.getByText("MEDIUM")).toBeInTheDocument();
    expect(screen.getByText("Balanced")).toBeInTheDocument();
    expect(screen.getByText("code_generation")).toBeInTheDocument();
    expect(screen.queryByText("Score")).not.toBeInTheDocument();
  });

  it.each([
    { threshold: 0, predicted_tier: "SIMPLE", expectedThreshold: "0.0%" },
    { threshold: 0.99, predicted_tier: "REASONING", expectedThreshold: "99.0%" },
  ])("keeps the prediction separate from an overridden tier at threshold $threshold", (scenario) => {
    render(
      <RoutingDecisionCard
        decision={{
          cause: "modality_escalation",
          tier: "REASONING",
          tier_label: "Vision",
          signals: ["modality:image"],
          heuristic_v2_forecast: {
            ...forecast,
            threshold: scenario.threshold,
            predicted_tier: scenario.predicted_tier,
          },
        }}
      />,
    );

    expect(screen.getByText("Vision")).toBeInTheDocument();
    expect(screen.getByText("Escalated for image input")).toBeInTheDocument();
    expect(screen.getByText("Predicted tier")).toBeInTheDocument();
    expect(screen.getByText(scenario.predicted_tier)).toBeInTheDocument();
    expect(screen.getByText(scenario.expectedThreshold)).toBeInTheDocument();
    expect(screen.getByText("modality:image")).toBeInTheDocument();
  });

  it.each([undefined, ["request-type:code_generation", "tier-probability:simple=0.100000"]])(
    "preserves legacy v2 rows without inventing a forecast when signals are %j",
    (signals) => {
      render(<RoutingDecisionCard decision={{ cause: "heuristic_v2", tier: "SIMPLE", signals }} />);

      expect(screen.getByText("Heuristic v2")).toBeInTheDocument();
      expect(screen.getByText("SIMPLE")).toBeInTheDocument();
      expect(screen.queryByText("Heuristic v2 estimates")).not.toBeInTheDocument();
      expect(screen.queryByText("Threshold")).not.toBeInTheDocument();
      for (const signal of signals ?? []) expect(screen.getByText(signal)).toBeInTheDocument();
    },
  );

  it.each(["capability_classifier", "modality_escalation"])(
    "shows the recorded Capability forecast for %s",
    (cause) => {
      render(
        <RoutingDecisionCard
          decision={{
            cause,
            tier: "COMPLEX",
            tier_label: "Deep",
            classifier_p_solve: 0,
            classifier_calibrated_p_solve: 0.864,
            classifier_threshold: 0.82,
            classifier_capability_boundary: "uncertain",
            classifier_primary_rule: "UNC-2",
            classifier_calibration_version: "calibration-1",
          }}
        />,
      );

      expect(screen.getByText("Capability estimates")).toBeInTheDocument();
      expect(screen.getByText("Efficient model solve chance")).toBeInTheDocument();
      expect(screen.getAllByText(/^\d+\.\d%$/).map((value) => value.textContent)).toEqual(["0.0%", "86.4%", "82.0%"]);
      expect(screen.getByText("Raw")).toBeInTheDocument();
      expect(screen.getByText("Calibrated")).toBeInTheDocument();
      expect(screen.getByText("Threshold")).toBeInTheDocument();
      expect(screen.getByText("uncertain")).toBeInTheDocument();
      expect(screen.getByText("UNC-2")).toBeInTheDocument();
      expect(screen.getByText("calibration-1")).toBeInTheDocument();
      expect(screen.getByText("Deep")).toBeInTheDocument();
      expect(screen.queryByText("FUSE v2 estimates")).not.toBeInTheDocument();
    },
  );

  it("omits absent Capability fields while preserving a recorded zero threshold", () => {
    render(
      <RoutingDecisionCard
        decision={{ cause: "capability_classifier", classifier_p_solve: 0.25, classifier_threshold: 0 }}
      />,
    );

    expect(screen.getAllByText(/^\d+\.\d%$/).map((value) => value.textContent)).toEqual(["25.0%", "0.0%"]);
    for (const label of ["Calibrated", "Calibration", "Boundary", "Rule"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });

  it.each(["llm_v2_classifier", "default_fallback"])(
    "shows the original calibrated FUSE v2 forecast for %s",
    (cause) => {
      render(
        <RoutingDecisionCard
          decision={{
            cause,
            routed_model: "fallback-model",
            classifier_efficient_p_solve: 0.25,
            classifier_capable_p_solve: 0.91,
            classifier_calibrated_efficient_p_solve: 0.75,
            classifier_calibrated_capable_p_solve: 0.8,
            classifier_max_quality_gap: 0.1,
            classifier_calibration_version: "calibration-2",
            signals: ["llm-v2:verification=tests"],
          }}
        />,
      );

      expect(screen.getByText("FUSE v2 estimates")).toBeInTheDocument();
      expect(screen.getAllByText(/^\d+\.\d%$/).map((value) => value.textContent)).toEqual([
        "25.0%",
        "91.0%",
        "75.0%",
        "80.0%",
      ]);
      expect(screen.getByText("Efficient (raw)")).toBeInTheDocument();
      expect(screen.getByText("Capable (raw)")).toBeInTheDocument();
      expect(screen.getByText("Efficient (calibrated)")).toBeInTheDocument();
      expect(screen.getByText("Capable (calibrated)")).toBeInTheDocument();
      expect(screen.getAllByText(/percentage points$/).map((value) => value.textContent)).toEqual([
        "5.0 percentage points",
        "10.0 percentage points",
      ]);
      expect(screen.getByText("Applied gap")).toBeInTheDocument();
      expect(screen.getByText("Allowed gap")).toBeInTheDocument();
      expect(screen.getByText("calibration-2")).toBeInTheDocument();
      expect(screen.getByText("llm-v2:verification=tests")).toBeInTheDocument();
      expect(screen.getByText("fallback-model")).toBeInTheDocument();
      expect(screen.queryByText("Capability estimates")).not.toBeInTheDocument();
    },
  );

  it("uses raw FUSE v2 probabilities without calibration and preserves negative and zero gaps", () => {
    render(
      <RoutingDecisionCard
        decision={{
          cause: "llm_v2_classifier",
          classifier_efficient_p_solve: 0.5,
          classifier_capable_p_solve: 0,
          classifier_max_quality_gap: 0,
        }}
      />,
    );

    expect(screen.getAllByText(/^\d+\.\d%$/).map((value) => value.textContent)).toEqual(["50.0%", "0.0%"]);
    expect(screen.getAllByText(/percentage points$/).map((value) => value.textContent)).toEqual([
      "-50.0 percentage points",
      "0.0 percentage points",
    ]);
    expect(screen.queryByText(/calibrated|Calibration/)).not.toBeInTheDocument();
  });

  it.each([
    { classifier_efficient_p_solve: 0, classifier_max_quality_gap: 0.2 },
    {
      classifier_efficient_p_solve: 0.4,
      classifier_capable_p_solve: 0.9,
      classifier_calibrated_efficient_p_solve: 0,
      classifier_calibration_version: "partial-calibration",
      classifier_max_quality_gap: 0.2,
    },
  ])("shows partial FUSE v2 estimates without inventing an applied gap: %j", (fields) => {
    render(<RoutingDecisionCard decision={{ cause: "llm_v2_classifier", ...fields }} />);

    expect(screen.getByText("FUSE v2 estimates")).toBeInTheDocument();
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.getByText("20.0 percentage points")).toBeInTheDocument();
    expect(screen.queryByText("Applied gap")).not.toBeInTheDocument();
    expect(screen.queryByText("Capable (calibrated)")).not.toBeInTheDocument();
  });

  it.each([
    ["capability_classifier", "Capability"],
    ["llm_v2_classifier", "FUSE v2"],
    ["capability_classifier_fallback", "Capable tier, Capability classifier failed"],
    ["llm_v2_fallback", "Capable tier, FUSE v2 classifier failed"],
    ["session_affinity_pin", "Pinned to session"],
  ])("labels %s without inventing a missing forecast", (cause, label) => {
    render(<RoutingDecisionCard decision={{ cause, tier: "MEDIUM" }} />);

    expect(screen.getByText(label)).toBeInTheDocument();
    expect(screen.getByText("MEDIUM")).toBeInTheDocument();
    expect(screen.queryByText(/estimates/)).not.toBeInTheDocument();
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
    expect(screen.getByText("Default model, classifier failed")).toBeInTheDocument();
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
    expect(screen.getByText("Fallback tier, classifier failed")).toBeInTheDocument();
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

  it("names the housekeeping sentinel so an operator can extend the pattern list", () => {
    // The sentinel is the string they would add to housekeeping_patterns to cover another
    // client, so the row is only useful if it says which one matched.
    render(
      <RoutingDecisionCard
        decision={{
          ...heuristic,
          cause: "housekeeping",
          matched_keyword: "Write the title in the predominant language of the session",
          score: undefined,
        }}
      />,
    );
    expect(
      screen.getByText('Client housekeeping call: "Write the title in the predominant language of the session"'),
    ).toBeInTheDocument();
  });

  it("still labels a housekeeping row when redaction dropped the sentinel", () => {
    // matched_keyword is prompt-quoting, so message-log redaction removes it. The row must
    // still read as a housekeeping decision rather than falling back to the raw cause.
    render(<RoutingDecisionCard decision={{ ...heuristic, cause: "housekeeping", score: undefined }} />);
    expect(screen.getByText("Client housekeeping call, classifier skipped")).toBeInTheDocument();
    expect(screen.queryByText("housekeeping")).not.toBeInTheDocument();
  });

  it("labels a modality pin override instead of showing the raw cause token", () => {
    render(<RoutingDecisionCard decision={{ ...heuristic, cause: "modality_pin_override" }} />);
    expect(screen.getByText("Overrode session pin for image input")).toBeInTheDocument();
    expect(screen.queryByText("modality_pin_override")).not.toBeInTheDocument();
  });

  it("labels a modality escalation instead of showing the raw cause token", () => {
    render(<RoutingDecisionCard decision={{ ...heuristic, cause: "modality_escalation" }} />);
    expect(screen.getByText("Escalated for image input")).toBeInTheDocument();
    expect(screen.queryByText("modality_escalation")).not.toBeInTheDocument();
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
