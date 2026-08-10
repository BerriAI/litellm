import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders } from "../../../tests/test-utils";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
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

const renderAs = (sessionRole: string) => {
  useAuthorizedMock.mockReturnValue({ userRole: sessionRole });
  return renderWithProviders(<SpendLogsTable {...defaultProps} userRole={sessionRole} />);
};

describe("SpendLogsTable", () => {
  beforeEach(() => {
    useAuthorizedMock.mockReturnValue({ userRole: "Admin" });
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

      expect(document.querySelector(".ant-spin")).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Request Logs" })).not.toBeInTheDocument();
    });

    it("renders the tabs (no spinner) once all credentials are present", () => {
      renderAs("Admin");

      expect(document.querySelector(".ant-spin")).not.toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Request Logs" })).toBeInTheDocument();
    });
  });
});
