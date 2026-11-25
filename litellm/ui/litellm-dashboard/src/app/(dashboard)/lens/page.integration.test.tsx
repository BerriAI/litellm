import { act, fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { setServerRootPath } from "@/lib/serverRootPath";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import type { CurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import LensPage from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const network = vi.fn<typeof fetch>();
const usersByBearer = new Map([
  ["Bearer sk-alice", "Alice"],
  ["Bearer sk-bob", "Bob"],
]);
const session = (user: string, expiresAt: number) => {
  const claims = {
    key: `sk-${user.toLowerCase()}`,
    auth_header_name: "X-Dashboard-Key",
    user_id: user,
    exp: expiresAt,
  };
  return `header.${btoa(JSON.stringify(claims))}.signature`;
};

function SessionControls() {
  const { token, authLoading, userID, setToken } = useAuth();
  if (authLoading) return <p>Loading session</p>;
  return (
    <>
      <p>{token ? `Signed in as ${userID}` : "Signed out"}</p>
      <button onClick={() => setToken(null)}>Remove session</button>
    </>
  );
}

const dashboard = (showLens = true) => (
  <AuthProvider>
    <SessionControls />
    {showLens && <LensPage />}
  </AuthProvider>
);

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
        return Response.json({ server_root_path: "/", proxy_base_url: "" });
      case "/lens/ui/":
        return new Response("<html></html>", { headers: { "X-Lens-UI-Auth": "bearer-v1" } });
      case "/lens/service":
        return Response.json({ configured: true, connected: true });
      case "/management/v1/users/me": {
        const bearer = request.headers.get("X-Dashboard-Key") ?? request.headers.get("Authorization");
        const userID = usersByBearer.get(bearer ?? "");
        if (!userID) return Response.json({ detail: "Invalid bearer" }, { status: 401 });
        const user: CurrentUser = { user_id: userID, user_email: null, user_role: "proxy_admin" };
        return Response.json(user);
      }
      default:
        throw new Error(`Unexpected request: ${url.pathname}`);
    }
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  clearTokenCookies();
  switchToWorkerUrl(null);
  setServerRootPath("/");
  setGlobalLitellmHeaderName("Authorization");
  vi.unstubAllGlobals();
});

it.each(["initial load", "shared cookie replacement"])(
  "uses the mounted Alice key and custom header on %s",
  async (step) => {
    const expiresAt = Math.floor(Date.now() / 1000) + 3600;
    document.cookie = `token=${session("Alice", expiresAt)}; Path=/`;
    const replacesCookie = step === "shared cookie replacement";
    const { rerender } = renderWithProviders(dashboard(!replacesCookie));

    if (replacesCookie) {
      expect(await screen.findByText("Signed in as Alice")).toBeVisible();
      document.cookie = `token=${session("Bob", expiresAt)}; Path=/`;
      rerender(dashboard());
    }

    const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
    expect(screen.getByText("Signed in as Alice")).toBeVisible();
    expect(getCookie("token")).toBe(session(replacesCookie ? "Bob" : "Alice", expiresAt));
    const requests = network.mock.calls.map(([input, init]) =>
      input instanceof Request ? input : new Request(new URL(String(input), window.location.href), init),
    );
    const serviceRequests = requests.filter((request) => new URL(request.url).pathname === "/lens/service");
    expect(serviceRequests[0].headers.get("X-Dashboard-Key")).toBe("Bearer sk-alice");
    expect(serviceRequests[0].headers.get("Authorization")).toBeNull();
    expect(serviceRequests).toHaveLength(1);

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
        command: { type: "auth", accessToken: "sk-alice", headerName: "X-Dashboard-Key" },
      },
      window.location.origin,
    );
  },
);

it("removes the frame when the mounted session token is removed", async () => {
  document.cookie = `token=${session("Alice", Math.floor(Date.now() / 1000) + 3600)}; Path=/`;
  renderWithProviders(dashboard());
  expect(await screen.findByTitle("Lens")).toBeVisible();

  fireEvent.click(screen.getByRole("button", { name: "Remove session" }));

  expect(screen.getByText("Signed out")).toBeVisible();
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});

it("removes the frame on rerender when the mounted token has expired", async () => {
  const expiresAt = Math.floor(Date.now() / 1000) + 3600;
  document.cookie = `token=${session("Alice", expiresAt)}; Path=/`;
  const { rerender } = renderWithProviders(dashboard());
  expect(await screen.findByTitle("Lens")).toBeVisible();

  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(expiresAt * 1000);
  rerender(dashboard());

  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});
