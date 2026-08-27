import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";

const { useAuthorizedMock, useOrganizationsMock } = vi.hoisted(() => ({
  useAuthorizedMock: vi.fn(),
  useOrganizationsMock: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: useOrganizationsMock,
}));

vi.mock("./RequestLogsPanel", () => ({
  default: function RequestLogsPanelMock() {
    return <div data-testid="request-logs-panel" />;
  },
}));

const fetchMock = vi.fn();

const jsonResponse = (body: unknown) => ({
  ok: true,
  status: 200,
  statusText: "OK",
  json: async () => body,
});

const requestedUrls = () => fetchMock.mock.calls.map(([url]) => String(url));

const emptyAuditLogs = { audit_logs: [], total: 0, page: 1, page_size: 50, total_pages: 0 };

const defaultProps = {
  accessToken: "sk-test",
  token: "jwt-test",
  userRole: "Admin",
  userID: "user-1",
  premiumUser: true,
};

const ORG_ADMIN_MEMBERSHIPS = [{ organization_id: "org-1", members: [{ user_id: "user-1", user_role: "org_admin" }] }];

const renderAs = (sessionRole: string, organizations: unknown[] = []) => {
  useAuthorizedMock.mockReturnValue({
    accessToken: "sk-test",
    userId: "user-1",
    userRole: sessionRole,
    premiumUser: true,
  });
  useOrganizationsMock.mockReturnValue({ data: organizations });
  return renderWithProviders(<SpendLogsTable {...defaultProps} userRole={sessionRole} />);
};

describe("SpendLogsTable network access by role", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    useOrganizationsMock.mockReturnValue({ data: [] });
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).includes("/audit")) {
        return jsonResponse(emptyAuditLogs);
      }
      if (String(url).includes("/v2/team/list")) {
        return jsonResponse({ teams: [] });
      }
      return jsonResponse({ keys: [], total_count: 0 });
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  it("fires neither the audit nor the deleted-teams request for an internal user", async () => {
    const user = userEvent.setup();
    renderAs("Internal User");

    // Liveness gate: the sibling Deleted Keys panel does reach the network, so a
    // silent absence below means the gate worked, not that nothing rendered.
    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/key/list"))).toBe(true));

    await user.click(screen.getByRole("tab", { name: "Deleted Keys" }));
    await user.click(screen.getByRole("tab", { name: "Request Logs" }));

    expect(requestedUrls().filter((url) => url.includes("/audit"))).toEqual([]);
    expect(requestedUrls().filter((url) => url.includes("/v2/team/list"))).toEqual([]);
  });

  it("fetches the deleted teams an org admin is entitled to, and still no audit logs", async () => {
    renderAs("Internal User", ORG_ADMIN_MEMBERSHIPS);

    await waitFor(() =>
      expect(requestedUrls().some((url) => url.includes("/v2/team/list") && url.includes("status=deleted"))).toBe(true),
    );

    expect(requestedUrls().filter((url) => url.includes("/audit"))).toEqual([]);
  });

  it("fetches deleted teams and audit logs for an admin", async () => {
    const user = userEvent.setup();
    renderAs("Admin");

    await waitFor(() =>
      expect(requestedUrls().some((url) => url.includes("/v2/team/list") && url.includes("status=deleted"))).toBe(true),
    );

    expect(requestedUrls().filter((url) => url.includes("/audit"))).toEqual([]);

    await user.click(screen.getByRole("tab", { name: "Audit Logs" }));

    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/audit"))).toBe(true));
  });

  it("leaves the audit request unsent when an admin selects a tab after Audit Logs", async () => {
    const user = userEvent.setup();
    renderAs("Admin");

    await user.click(screen.getByRole("tab", { name: "Deleted Teams" }));

    expect(screen.getByRole("tab", { name: "Deleted Teams" })).toHaveAttribute("aria-selected", "true");
    expect(requestedUrls().filter((url) => url.includes("/audit"))).toEqual([]);

    await user.click(screen.getByRole("tab", { name: "Audit Logs" }));

    await waitFor(() => expect(requestedUrls().some((url) => url.includes("/audit"))).toBe(true));
  });
});
