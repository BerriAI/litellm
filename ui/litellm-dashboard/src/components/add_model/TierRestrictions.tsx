import React from "react";
import { CUSTOM_TIER_RESTRICTIONS, CustomTierSet, TierRestriction } from "./tier_rows";

export const restrictedBy = (
  value: { custom_tier_set?: CustomTierSet; classifier_type?: string },
  key: keyof typeof CUSTOM_TIER_RESTRICTIONS,
): TierRestriction | undefined => {
  if (value.custom_tier_set) return CUSTOM_TIER_RESTRICTIONS[key];
  if (value.classifier_type === "llm_v2" && key === "adaptive")
    return {
      omit: ["adaptive", "adaptive_weights", "adaptive_eligible", "tier_distance_penalty"],
      reason: "Fuse v2 uses its quality-gap decision directly; adaptive routing is unavailable",
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
