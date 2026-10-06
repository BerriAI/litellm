"use client";

import { createContext, useContext } from "react";
import type { TraceSummary } from "@/components/lens/traces/types";

export interface Onboarding {
  readonly readOnly: boolean;
  readonly canViewInvestigations: boolean;
  readonly canInvestigate: boolean;
  readonly canMintTracingKey: boolean;
  connect(): void;
  create(): void;
  openTrace(trace: TraceSummary): void;
}

const OnboardingContext = createContext<Onboarding | null>(null);

export const OnboardingProvider = OnboardingContext.Provider;

export function useOnboarding(): Onboarding {
  const onboarding = useContext(OnboardingContext);
  if (!onboarding) throw new Error("useOnboarding needs an OnboardingProvider above it");
  return onboarding;
}
