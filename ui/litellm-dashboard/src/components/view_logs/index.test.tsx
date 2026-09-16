import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";
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
  default: function RequestLogsPanelMock({ isActive }: { isActive: boolean }) {
    return <div data-testid="request-logs-panel">{isActive ? "active" : "inactive"}</div>;
  },
}));

vi.mock("./AuditLogsPanel", () => ({
  default: function AuditLogsPanelMock({ isActive }: { isActive: boolean }) {
    return <div data-testid="audit-logs-panel">{isActive ? "active" : "inactive"}</div>;
  },
}));

vi.mock("../DeletedKeysPage/DeletedKeysPage", () => ({
  default: function DeletedKeysPageMock() {
    return <div data-testid="deleted-keys-page" />;
  },
}));

vi.mock("../DeletedTeamsPage/DeletedTeamsPage", () => ({
  default: function DeletedTeamsPageMock() {
    return <div data-testid="deleted-teams-page" />;
  },
}));

const defaultProps = {
  accessToken: "test-token",
  token: "test-token",
  userRole: "Admin",
  userID: "user-1",
  premiumUser: false,
};

const ORG_ADMIN_MEMBERSHIPS = [{ organization_id: "org-1", members: [{ user_id: "user-1", user_role: "org_admin" }] }];

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderAs = (sessionRole: string, organizations: unknown[] = [], urlOptions: UrlOptions = {}) => {
  useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole: sessionRole });
  useOrganizationsMock.mockReturnValue({ data: organizations });
  return renderWithProviders(<SpendLogsTable {...defaultProps} userRole={sessionRole} />, urlOptions);
};

const renderKeepingMountUrlWrites = (sessionRole: string, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) => {
  useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole: sessionRole });
  useOrganizationsMock.mockReturnValue({ data: [] });
  return render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>
        <SpendLogsTable {...defaultProps} userRole={sessionRole} />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );
};

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

const tabNames = () => screen.getAllByRole("tab").map((tab) => tab.textContent);

describe("SpendLogsTable", () => {
  beforeEach(() => {
    useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole: "Admin" });
    useOrganizationsMock.mockReturnValue({ data: [] });
  });

  it("renders the four log tabs", () => {
    renderAs("Admin");

    for (const label of ["Request Logs", "Audit Logs", "Deleted Keys", "Deleted Teams"]) {
      expect(screen.getByRole("tab", { name: label })).toBeInTheDocument();
    }
  });

  it("marks only the visible tab's panel active so background tabs do not query", async () => {
    const user = userEvent.setup();
    renderAs("Admin");

    expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("active");

    await user.click(screen.getByRole("tab", { name: "Audit Logs" }));

    expect(await screen.findByTestId("audit-logs-panel")).toHaveTextContent("active");
    expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");
  });

  describe("admin-only tabs", () => {
    it.each(["Internal User", "Internal Viewer"])("hides Audit Logs and Deleted Teams from %s", (role) => {
      renderAs(role);

      expect(screen.getByRole("tab", { name: "Request Logs" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Deleted Keys" })).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Audit Logs" })).not.toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Deleted Teams" })).not.toBeInTheDocument();
    });

    it("never mounts the panels that call the admin-only endpoints for an internal user", () => {
      renderAs("Internal User");

      expect(screen.queryByTestId("audit-logs-panel")).not.toBeInTheDocument();
      expect(screen.queryByTestId("deleted-teams-page")).not.toBeInTheDocument();
      expect(screen.getByTestId("deleted-keys-page")).toBeInTheDocument();
    });
  });

  describe("organization admins", () => {
    it("shows Deleted Teams to an org admin, whose session role reads as a plain internal user", () => {
      renderAs("Internal User", ORG_ADMIN_MEMBERSHIPS);

      expect(screen.getByRole("tab", { name: "Deleted Teams" })).toBeInTheDocument();
      expect(screen.getByTestId("deleted-teams-page")).toBeInTheDocument();
    });

    it("does not hand an org admin the Audit Logs tab, which the backend still refuses them", () => {
      renderAs("Internal User", ORG_ADMIN_MEMBERSHIPS);

      expect(tabNames()).toEqual(["Request Logs", "Deleted Keys", "Deleted Teams"]);
      expect(screen.queryByTestId("audit-logs-panel")).not.toBeInTheDocument();
    });

    it("keeps an internal user in the same org without an org_admin membership at two tabs", () => {
      renderAs("Internal User", [
        { organization_id: "org-1", members: [{ user_id: "user-1", user_role: "internal_user" }] },
      ]);

      expect(tabNames()).toEqual(["Request Logs", "Deleted Keys"]);
    });

    it("activates the org admin's selected tab rather than the one at the four-tab index", async () => {
      const user = userEvent.setup();
      renderAs("Internal User", ORG_ADMIN_MEMBERSHIPS);

      await user.click(screen.getByRole("tab", { name: "Deleted Teams" }));

      expect(screen.getByRole("tab", { name: "Deleted Teams" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");

      await user.click(screen.getByRole("tab", { name: "Request Logs" }));

      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("active");
    });
  });

  describe("tab index mapping", () => {
    it("activates the panel the admin selected, not the one at the old hardcoded index", async () => {
      const user = userEvent.setup();
      renderAs("Admin");

      await user.click(screen.getByRole("tab", { name: "Deleted Keys" }));

      expect(screen.getByTestId("audit-logs-panel")).toHaveTextContent("inactive");
      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");
    });

    it("keeps the audit panel inert when an admin selects the last tab", async () => {
      const user = userEvent.setup();
      renderAs("Admin");

      await user.click(screen.getByRole("tab", { name: "Deleted Teams" }));

      expect(screen.getByTestId("audit-logs-panel")).toHaveTextContent("inactive");
      expect(screen.getByTestId("deleted-teams-page")).toBeInTheDocument();
    });

    it("selects the last visible tab for an internal user and returns to Request Logs", async () => {
      const user = userEvent.setup();
      renderAs("Internal User");

      await user.click(screen.getByRole("tab", { name: "Deleted Keys" }));

      expect(screen.getByTestId("deleted-keys-page")).toBeInTheDocument();
      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");

      await user.click(screen.getByRole("tab", { name: "Request Logs" }));

      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("active");
    });
  });

  describe("auth-not-ready guard", () => {
    it("shows a loading spinner when credentials are not yet resolved", () => {
      useAuthorizedMock.mockReturnValue({ userRole: "Admin" });
      renderWithProviders(<SpendLogsTable {...defaultProps} accessToken={null} />);

      expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Request Logs" })).not.toBeInTheDocument();
    });

    it("renders the tabs (no spinner) once all credentials are present", () => {
      renderAs("Admin");

      expect(document.querySelector('[aria-busy="true"]')).not.toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Request Logs" })).toBeInTheDocument();
    });
  });

  describe("URL tab state", () => {
    it("opens the tab named by ?tab= on load", () => {
      renderAs("Admin", [], { searchParams: "?tab=audit-logs" });

      expect(screen.getByRole("tab", { name: "Audit Logs" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByTestId("audit-logs-panel")).toHaveTextContent("active");
      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");
    });

    it("writes the selected tab's slug to ?tab=", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderAs("Admin", [], { onUrlUpdate });

      await user.click(screen.getByRole("tab", { name: "Deleted Keys" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("tab")).toBe("deleted-keys"));
    });

    it("drops ?tab= when the user returns to Request Logs", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderAs("Admin", [], { searchParams: "?tab=deleted-teams", onUrlUpdate });
      expect(screen.getByTestId("deleted-teams-page")).toBeInTheDocument();

      await user.click(screen.getByRole("tab", { name: "Request Logs" }));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.has("tab")).toBe(false);
      expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("active");
    });

    it("falls back to Request Logs and clears an unknown ?tab=", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountUrlWrites("Admin", "?tab=nonsense&log_id=abc", onUrlUpdate);

      expect(screen.getByRole("tab", { name: "Request Logs" })).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.has("tab")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.get("log_id")).toBe("abc");
    });

    it("falls back and clears ?tab=audit-logs for an internal user who cannot see that tab", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountUrlWrites("Internal User", "?tab=audit-logs", onUrlUpdate);

      expect(screen.getByRole("tab", { name: "Request Logs" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByTestId("audit-logs-panel")).not.toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.has("tab")).toBe(false);
    });

    it("lets an org admin deep-link to Deleted Teams even though the session role reads as internal user", () => {
      renderAs("Internal User", ORG_ADMIN_MEMBERSHIPS, { searchParams: "?tab=deleted-teams" });

      expect(screen.getByRole("tab", { name: "Deleted Teams" })).toHaveAttribute("aria-selected", "true");
    });

    it("keeps ?tab=audit-logs while credentials resolve, then lands on it once the admin role is known", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      useAuthorizedMock.mockReturnValue({ userId: null, userRole: null });
      const { rerender } = renderWithProviders(
        <SpendLogsTable {...defaultProps} accessToken={null} userRole={null} userID={null} />,
        { searchParams: "?tab=audit-logs", onUrlUpdate },
      );
      expect(screen.getByRole("status", { name: "Loading" })).toBeInTheDocument();

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(onUrlUpdate).not.toHaveBeenCalled();

      useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole: "Admin" });
      rerender(<SpendLogsTable {...defaultProps} />);

      expect(screen.getByRole("tab", { name: "Audit Logs" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByTestId("audit-logs-panel")).toHaveTextContent("active");
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });
  });
});
