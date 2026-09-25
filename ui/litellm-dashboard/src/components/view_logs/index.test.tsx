import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders } from "../../../tests/test-utils";

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

const renderAs = (sessionRole: string, organizations: unknown[] = []) => {
  useAuthorizedMock.mockReturnValue({ userId: "user-1", userRole: sessionRole });
  useOrganizationsMock.mockReturnValue({ data: organizations });
  return renderWithProviders(<SpendLogsTable {...defaultProps} userRole={sessionRole} />);
};

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
});
