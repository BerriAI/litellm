import type { ReactElement } from "react";
import { LensServicesProvider, liveLensServices } from "@/components/lens/data/LensServices";
import { LENS_INTRO_DISMISSED } from "@/components/lens/storage";
import { writeStorage } from "@/lib/storage";
import { renderWithProviders } from "./test-utils";

/** The Lens introduction opens on a first visit; tests about anything else start with it dismissed. */
export function dismissLensIntro() {
  writeStorage(LENS_INTRO_DISMISSED, true);
}

type LensRenderOptions = Parameters<typeof renderWithProviders>[1] & { accessToken?: string };

/** Lens components read their API from context, so standalone renders need the live services for a token. */
export function renderWithLens(ui: ReactElement, options?: LensRenderOptions) {
  const { accessToken = "test", ...renderOptions } = options ?? {};
  const services = liveLensServices(accessToken);
  const wrap = (element: ReactElement) => <LensServicesProvider services={services}>{element}</LensServicesProvider>;
  const view = renderWithProviders(wrap(ui), renderOptions);
  return { ...view, rerender: (next: ReactElement) => view.rerender(wrap(next)) };
}
