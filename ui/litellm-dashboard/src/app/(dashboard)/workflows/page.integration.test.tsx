import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Workflows from "./page";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

const fetchMock = vi.fn();

const requestedUrls = () => fetchMock.mock.calls.map(([url]) => String(url));

const renderAs = (userRole: string) => {
  useAuthorizedMock.mockReturnValue({ accessToken: "sk-test", userId: "u1", userRole });
  return renderWithProviders(<Workflows />);
};

// Deep-linking to /ui/workflows bypasses the sidebar, so the page itself has to
// refuse the render. `/v1/workflows/runs` is proxy-admin-only, so any request
// from a non-admin is the 401 this gate exists to stop.
describe("Workflows page access by role", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({ runs: [], count: 0 }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  it("lists workflow runs for an admin", async () => {
    renderAs("Admin");

    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/v1/workflows/runs"))).toBe(true));
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin", "Unknown Role"])(
    "renders the admin-only notice and fires no workflow request for %s",
    async (userRole) => {
      renderAs(userRole);

      expect(await screen.findByText("Workflow Runs is only available to admin users.")).toBeInTheDocument();
      await waitFor(() => expect(requestedUrls().filter((url) => url.includes("/v1/workflows"))).toEqual([]));
    },
  );

  it("hides the deprecation banner along with the page body for a denied role", () => {
    renderAs("Internal User");

    expect(screen.queryByText(/draft deprecation list/i)).not.toBeInTheDocument();
  });
});
