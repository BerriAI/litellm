import type { ReactElement } from "react";
import { LensServicesProvider, liveLensServices } from "@/components/lens/LensServices";
import { renderWithProviders } from "./test-utils";

type LensRenderOptions = Parameters<typeof renderWithProviders>[1] & { accessToken?: string };

/** Lens components read their API from context, so standalone renders need the live services for a token. */
export function renderWithLens(ui: ReactElement, options?: LensRenderOptions) {
  const { accessToken = "test", ...renderOptions } = options ?? {};
  const services = liveLensServices(accessToken);
  const wrap = (element: ReactElement) => <LensServicesProvider services={services}>{element}</LensServicesProvider>;
  const view = renderWithProviders(wrap(ui), renderOptions);
  return { ...view, rerender: (next: ReactElement) => view.rerender(wrap(next)) };
}
