import type { ReactElement } from "react";
import { vi } from "vitest";
import { LensServicesProvider, liveLensServices } from "@/components/lens/data/LensServices";
import { OnboardingProvider, type Onboarding } from "@/components/lens/onboarding/OnboardingContext";
import { renderWithProviders } from "./test-utils";

export interface StubbedRequest {
  readonly path: string;
  readonly method: string;
  readonly query: URLSearchParams;
  readonly body: unknown;
  readonly authorization: string | null;
}

/** The path of a stubbed fetch call, whether a client passed a URL or a Request. */
export const requestPath = (input: RequestInfo | URL): string =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;

/** Reads a stubbed fetch call the same way whether a client passed a URL and init or a Request. */
export async function readRequest(input: RequestInfo | URL, init?: RequestInit): Promise<StubbedRequest> {
  const request =
    input instanceof Request
      ? input
      : new Request(new URL(String(input), "http://localhost"), { ...init, signal: undefined });
  const url = new URL(request.url);
  const text = await request.clone().text();
  return {
    path: url.pathname,
    method: request.method,
    query: url.searchParams,
    body: text ? JSON.parse(text) : undefined,
    authorization: request.headers.get("Authorization"),
  };
}

export interface GatewayRequest {
  readonly query: Readonly<Record<string, string>>;
  readonly body: unknown;
  readonly authorization: string | null;
}

type GatewayHandler = (path: string, request: GatewayRequest) => unknown;

/**
 * Fakes the proxy at fetch, so Lens code runs its real HTTP clients. Each method's handler gets the path and the
 * parsed query and body; a rejection becomes a 500 carrying its message, and a pending promise stays pending.
 */
export function stubGateway() {
  const gateway = {
    get: vi.fn<GatewayHandler>(),
    post: vi.fn<GatewayHandler>(),
    put: vi.fn<GatewayHandler>(),
    patch: vi.fn<GatewayHandler>(),
    delete: vi.fn<GatewayHandler>(),
  };
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const { path, method, query, body, authorization } = await readRequest(input, init);
      const handler = gateway[method.toLowerCase() as keyof typeof gateway];
      try {
        return Response.json((await handler(path, { query: Object.fromEntries(query), body, authorization })) ?? null);
      } catch (error) {
        return Response.json({ detail: error instanceof Error ? error.message : String(error) }, { status: 500 });
      }
    }),
  );
  return gateway;
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
