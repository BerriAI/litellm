import React from "react";
import { CUSTOM_TIER_RESTRICTIONS, CustomTierSet, TierRestriction } from "./tier_rows";

export const restrictedBy = (
  value: { custom_tier_set?: CustomTierSet },
  key: keyof typeof CUSTOM_TIER_RESTRICTIONS,
): TierRestriction | undefined => (value.custom_tier_set ? CUSTOM_TIER_RESTRICTIONS[key] : undefined);

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
