import React from "react";

import { type ComplexityRouterConfigValue, heuristicScoringRole, usesLlmClassifier } from "./ComplexityRouterConfig";
import { restrictedBy } from "./TierRestrictions";

const tierConfigIntroText = (value: ComplexityRouterConfigValue): string => {
  if (value.classifier_type === "heuristic_v2") {
    return "The complexity router classifies each request with a calibrated local four-tier model (no API calls). Configure which model(s) handle each tier.";
  }
  if (heuristicScoringRole(value) === "never") {
    return "The complexity router classifies each request with your classifier model and routes it to that tier. Configure which model(s) handle each tier.";
  }
  return "The complexity router automatically classifies requests by complexity using rule-based scoring (no API calls, <1ms latency). Configure which model(s) handle each tier.";
};

const TierConfigIntro: React.FC<{ value: ComplexityRouterConfigValue }> = ({ value }) => (
  <>
    <span className="block mb-6 text-muted-foreground">{tierConfigIntroText(value)}</span>

    <span className="block mb-4 text-xs text-muted-foreground">
      {restrictedBy(value, "displayNames")?.reason ??
        "Rename a tier to use your own vocabulary in the dashboard and your spend logs. Renaming doesn't change how requests are classified, and callers never see these names."}
      {!value.custom_tier_set &&
        usesLlmClassifier(value.classifier_type) &&
        " Your classifier model reads these names, so clearer ones can sharpen its choices."}
    </span>
  </>
);

export default TierConfigIntro;
