import React from "react";
import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import { CUSTOM_TIER_RESTRICTIONS, CustomTierSet, TierRestriction } from "./tier_rows";

export const restrictedBy = (
  value: { custom_tier_set?: CustomTierSet; classifier_type?: string },
  key: keyof typeof CUSTOM_TIER_RESTRICTIONS,
  t: TFunction = i18n.t.bind(i18n),
): TierRestriction | undefined => {
  if (value.custom_tier_set)
    return { omit: CUSTOM_TIER_RESTRICTIONS[key].omit, reason: t(CUSTOM_TIER_RESTRICTIONS[key].reason) };
  if (value.classifier_type === "llm_v2" && key === "adaptive")
    return {
      omit: ["adaptive", "adaptive_weights", "adaptive_eligible", "tier_distance_penalty"],
      reason: t("addModel.tierRestrictions.fuseAdaptiveUnavailable"),
    };
  return undefined;
};

export const Restricted: React.FC<{ by: TierRestriction | undefined; children: React.ReactNode }> = ({
  by,
  children,
}) => (by ? <span className="block text-sm text-muted-foreground">{by.reason}</span> : <>{children}</>);

/** A labelled section whose body is replaced by the reason an edited tier set forbids it. */
export const RestrictedSection: React.FC<{
  heading: string;
  by: TierRestriction | undefined;
  children: React.ReactNode;
}> = ({ heading, by, children }) => (
  <div>
    <strong className="block mb-1 font-semibold">{heading}</strong>
    {by ? <span className="block text-sm text-muted-foreground">{by.reason}</span> : children}
  </div>
);
