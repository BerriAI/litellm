import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { AuthProvider } from "@/contexts/AuthContext";
import SidebarAccountMenu from "@/components/SidebarAccountMenu/SidebarAccountMenu";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { setServerRootPath } from "@/lib/serverRootPath";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import type { CurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import LensPage from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

const network = vi.fn<typeof fetch>();
const logout = vi.fn();
const usersByBearer = new Map<string, CurrentUser>([
  ["Bearer sk-alice", { user_id: "Alice", user_email: null, user_role: "proxy_admin" }],
  ["Bearer sk-bob", { user_id: "Bob", user_email: null, user_role: "internal_user" }],
]);
const session = (user: string, expiresAt: number) => {
  const claims = {
    key: `sk-${user.toLowerCase()}`,
    auth_header_name: user === "Alice" ? "X-Admin-Key" : "X-Internal-Key",
    user_id: user,
    user_role: user === "Alice" ? "proxy_admin" : "internal_user",
    exp: expiresAt,
  };
  return `header.${btoa(JSON.stringify(claims))}.signature`;
};

const dashboard = (showLens = true) => (
  <AuthProvider>
    <SidebarAccountMenu onLogout={logout} />
    {showLens && <LensPage />}
  </AuthProvider>
);

const serviceRequests = () =>
  network.mock.calls
    .map(([input, init]) =>
      input instanceof Request ? input : new Request(new URL(String(input), window.location.href), init),
    )
    .filter((request) => new URL(request.url).pathname === "/lens/service");

function expectLensCredentials(iframe: HTMLIFrameElement, accessToken: string, headerName: string) {
  const child = iframe.contentWindow!;
  const postMessage = vi.spyOn(child, "postMessage").mockImplementation(() => {});
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        source: child,
        origin: window.location.origin,
        data: { channel: "litellm-lens:v1", type: "auth-ready" },
      }),
    );
  });
  expect(postMessage).toHaveBeenCalledExactlyOnceWith(
    {
      channel: "litellm-lens:v1",
      command: { type: "auth", accessToken, headerName },
    },
    window.location.origin,
  );
  const request = serviceRequests().at(-1);
  expect(request?.headers.get(headerName)).toBe(`Bearer ${accessToken}`);
  expect(request?.headers.get("Authorization")).toBeNull();
}

beforeEach(() => {
  network.mockReset();
  testQueryClient.clear();
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
  clearTokenCookies();
  setGlobalLitellmHeaderName("Authorization");
  vi.stubGlobal("fetch", network);
  network.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), window.location.href), init);
    const url = new URL(request.url);
    switch (url.pathname) {
      case "/ui/litellm-runtime-config.json":
        return Response.json({});
      case "/litellm/.well-known/litellm-ui-config":
        return Response.json({ server_root_path: "/", proxy_base_url: "", admin_ui_disabled: false });
      case "/health/readiness/details":
        return Response.json({ status: "healthy" });
      case "/lens/ui/":
        return new Response("<html></html>", { headers: { "X-Lens-UI-Auth": "bearer-v1" } });
      case "/lens/service":
        return Response.json({ configured: true, connected: true });
      case "/management/v1/users/me": {
        const bearer =
          request.headers.get("X-Admin-Key") ??
          request.headers.get("X-Internal-Key") ??
          request.headers.get("Authorization");
        const user = usersByBearer.get(bearer ?? "");
        if (!user) return Response.json({ detail: "Invalid bearer" }, { status: 401 });
        return Response.json(user);
      }
      default:
        throw new Error(`Unexpected request: ${url.pathname}`);
    }
  });
});

afterEach(() => {
  testQueryClient.clear();
  vi.useRealTimers();
  vi.restoreAllMocks();
  clearTokenCookies();
  switchToWorkerUrl(null);
  setServerRootPath("/");
  setGlobalLitellmHeaderName("Authorization");
  vi.unstubAllGlobals();
});

it("uses the visible Admin account and JWT header for the first service request and handshake", async () => {
  document.cookie = `token=${session("Alice", Math.floor(Date.now() / 1000) + 3600)}; Path=/`;
  renderWithProviders(dashboard());
  expect(await screen.findByRole("button", { name: /Account menu.*Admin.*signed in as Alice/ })).toBeVisible();
  const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
  expectLensCredentials(iframe, "sk-alice", "X-Admin-Key");
  expect(serviceRequests()).toHaveLength(1);
});

it.each(["navigation to Lens", "a shared rerender with Lens open"])(
  "matches the visible Internal User after another login and %s",
  async (step) => {
    const expiresAt = Math.floor(Date.now() / 1000) + 3600;
    const alice = session("Alice", expiresAt);
    document.cookie = `token=${alice}; Path=/`;
    const wasOpen = step === "a shared rerender with Lens open";
    const { rerender } = renderWithProviders(dashboard(wasOpen));
    expect(await screen.findByRole("button", { name: /Account menu.*Admin.*signed in as Alice/ })).toBeVisible();
    const previousFrame = wasOpen ? await screen.findByTitle<HTMLIFrameElement>("Lens") : null;
    if (previousFrame) expectLensCredentials(previousFrame, "sk-alice", "X-Admin-Key");

    clearTokenCookies();
    const bob = session("Bob", expiresAt);
    document.cookie = `token=${bob}; Path=/`;
    rerender(dashboard());

    expect(await screen.findByRole("button", { name: /Account menu.*Internal User.*signed in as Bob/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /Account menu.*Admin.*signed in as Alice/ })).not.toBeInTheDocument();
    const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
    expectLensCredentials(iframe, "sk-bob", "X-Internal-Key");
    expect(serviceRequests()).toHaveLength(wasOpen ? 2 : 1);
    if (previousFrame) expect(previousFrame).not.toBeInTheDocument();
    expect(getCookie("token")).toBe(bob);
  },
);

it.each(["missing", "another account"])(
  "keeps Lens on the visible dashboard account when the root cookie is %s",
  async (rootCookie) => {
    const expiresAt = Math.floor(Date.now() / 1000) + 3600;
    const alice = session("Alice", expiresAt);
    document.cookie = `token=${alice}; Path=/ui`;
    document.cookie = `token=${alice}; Path=/`;
    const { rerender } = renderWithProviders(dashboard(false));
    expect(await screen.findByRole("button", { name: /Account menu.*Admin.*signed in as Alice/ })).toBeVisible();

    document.cookie =
      rootCookie === "missing" ? "token=; Max-Age=0; Path=/" : `token=${session("Bob", expiresAt)}; Path=/`;
    rerender(dashboard());

    expect(await screen.findByRole("button", { name: /Account menu.*Admin.*signed in as Alice/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /Account menu.*signed in as Bob/ })).not.toBeInTheDocument();
    const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
    expectLensCredentials(iframe, "sk-alice", "X-Admin-Key");
    expect(serviceRequests()).toHaveLength(1);
  },
);

it("removes the frame and account identity when dashboard credentials are cleared", async () => {
  document.cookie = `token=${session("Alice", Math.floor(Date.now() / 1000) + 3600)}; Path=/`;
  const { rerender } = renderWithProviders(dashboard());
  expect(await screen.findByTitle("Lens")).toBeVisible();

  vi.stubGlobal("location", { ...window.location, replace: vi.fn() });
  clearTokenCookies();
  rerender(dashboard());

  expect(screen.queryByRole("button", { name: /Account menu.*signed in as Alice/ })).not.toBeInTheDocument();
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});

it("redirects an initially expired dashboard session to login without mounting Lens", async () => {
  const replace = vi.fn();
  vi.stubGlobal("location", { ...window.location, replace });
  const expiresAt = Math.floor(Date.now() / 1000) - 1;
  document.cookie = `token=${session("Alice", expiresAt)}; Path=/`;
  renderWithProviders(dashboard());

  await waitFor(() => expect(replace).toHaveBeenCalledWith(expect.stringContaining("/ui/login/")));
  expect(getCookie("token")).toBeNull();
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
  expect(serviceRequests()).toHaveLength(0);
});
