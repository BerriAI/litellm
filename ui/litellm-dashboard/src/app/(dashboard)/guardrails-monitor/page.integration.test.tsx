import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsMonitor from "./page";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

const fetchMock = vi.fn();

const requestedUrls = () => fetchMock.mock.calls.map(([url]) => String(url));

const renderAs = (userRole: string) => {
  useAuthorizedMock.mockReturnValue({ accessToken: "sk-test", userId: "u1", userRole });
  return renderWithProviders(<GuardrailsMonitor />);
};

// `/guardrails/usage/*` aggregates across tenants and is listed in
// admin_viewer_routes, so it is proxy-admin-only. Nothing on this page works
// for a non-admin, hence the whole page is gated rather than a section of it.
describe("Guardrails Monitor page access by role", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({ rows: [], chart: [], totalRequests: 0, totalBlocked: 0, passRate: 100 }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  it("fetches the guardrails usage overview for an admin", async () => {
    renderAs("Admin");

    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/guardrails/usage/overview"))).toBe(true));
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin", "Unknown Role"])(
    "renders the admin-only notice and fires no usage request for %s",
    async (userRole) => {
      renderAs(userRole);

      expect(await screen.findByText("Guardrails Monitor is only available to admin users.")).toBeInTheDocument();
      await waitFor(() => expect(requestedUrls().filter((url) => url.includes("/guardrails/usage"))).toEqual([]));
    },
  );
});
