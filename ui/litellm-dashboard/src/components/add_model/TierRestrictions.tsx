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
