import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AuthProvider } from "@/contexts/AuthContext";
import Layout from "./layout";

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: vi.fn(), replace: vi.fn() })),
  useSearchParams: vi.fn(() => new URLSearchParams()),
  usePathname: vi.fn(() => "/ui/guardrails"),
}));

vi.mock("@/components/navbar", () => ({
  default: () => <div data-testid="navbar" />,
}));

vi.mock("@/app/(dashboard)/components/SidebarProvider", () => ({
  default: () => <div data-testid="sidebar" />,
}));

vi.mock("@/components/DebugWarningBanner", () => ({
  DebugWarningBanner: () => null,
}));

vi.mock("@/contexts/ThemeContext", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/common_components/LoadingScreen", () => ({
  default: () => <div data-testid="loading-screen" />,
}));

const deferred = (() => {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
})();

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getUiConfig: vi.fn(() => deferred.promise),
    setGlobalLitellmHeaderName: vi.fn(),
  };
});

describe("(dashboard) Layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not mount route content until getUiConfig has resolved", async () => {
    render(
      <AuthProvider>
        <Layout>
          <div data-testid="page-content" />
        </Layout>
      </AuthProvider>,
    );

    // While getUiConfig is in flight, proxyBaseUrl/serverRootPath still hold their
    // module defaults; mounting pages now would fire fetches without the
    // SERVER_ROOT_PATH prefix. The layout must hold them back.
    await waitFor(() => expect(screen.getByTestId("loading-screen")).toBeTruthy());
    expect(screen.queryByTestId("page-content")).toBeNull();
    expect(screen.queryByTestId("navbar")).toBeNull();

    deferred.resolve();

    await waitFor(() => expect(screen.getByTestId("page-content")).toBeTruthy());
    expect(screen.getByTestId("navbar")).toBeTruthy();
    expect(screen.queryByTestId("loading-screen")).toBeNull();
  });
});
