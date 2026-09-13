import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Memory from "./page";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

const fetchMock = vi.fn();

const requestedUrls = () => fetchMock.mock.calls.map(([url]) => String(url));

const renderAs = (userRole: string) => {
  useAuthorizedMock.mockReturnValue({ accessToken: "sk-test", userId: "u1", userRole });
  return renderWithProviders(<Memory />);
};

// `/v1/memory` scopes rows per caller in the handler, but the route gate keeps
// it proxy-admin-only, so a non-admin deep-linking to /ui/memory gets a 401.
describe("Memory page access by role", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      text: async () => "",
      json: async () => ({ memories: [], total: 0 }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  it("lists memory entries for an admin", async () => {
    renderAs("Admin");

    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/v1/memory"))).toBe(true));
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin", "Unknown Role"])(
    "renders the admin-only notice and fires no memory request for %s",
    async (userRole) => {
      renderAs(userRole);

      expect(await screen.findByText("Memory is only available to admin users.")).toBeInTheDocument();
      await waitFor(() => expect(requestedUrls().filter((url) => url.includes("/v1/memory"))).toEqual([]));
    },
  );

  it("hides the deprecation banner along with the page body for a denied role", () => {
    renderAs("Internal User");

    expect(screen.queryByText(/draft deprecation list/i)).not.toBeInTheDocument();
  });
});
