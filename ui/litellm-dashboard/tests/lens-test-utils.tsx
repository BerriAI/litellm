import type { ReactElement } from "react";
import { LensServicesProvider, liveLensServices } from "@/components/lens/data/LensServices";
import { OnboardingProvider, type Onboarding } from "@/components/lens/onboarding/OnboardingContext";
import { LENS_INTRO_DISMISSED } from "@/components/lens/storage";
import { writeStorage } from "@/lib/storage";
import { renderWithProviders } from "./test-utils";

/** The Lens introduction opens on a first visit; tests about anything else start with it dismissed. */
export function dismissLensIntro() {
  writeStorage(LENS_INTRO_DISMISSED, true);
}

type LensRenderOptions = Parameters<typeof renderWithProviders>[1] & {
  accessToken?: string;
  onboarding?: Partial<Onboarding>;
};

/** Lens components read their API from context, so standalone renders need the live services for a token. */
export function renderWithLens(ui: ReactElement, options?: LensRenderOptions) {
  const { accessToken = "test", onboarding: overrides, ...renderOptions } = options ?? {};
  const services = liveLensServices(accessToken);
  const onboarding: Onboarding = {
    readOnly: false,
    canViewInvestigations: true,
    canInvestigate: true,
    canMintTracingKey: true,
    connect: () => {},
    create: () => {},
    openTrace: () => {},
    ...overrides,
  };
  const wrap = (element: ReactElement) => (
    <LensServicesProvider services={services}>
      <OnboardingProvider value={onboarding}>{element}</OnboardingProvider>
    </LensServicesProvider>
  );
  const view = renderWithProviders(wrap(ui), renderOptions);
  return { ...view, rerender: (next: ReactElement) => view.rerender(wrap(next)) };
}
