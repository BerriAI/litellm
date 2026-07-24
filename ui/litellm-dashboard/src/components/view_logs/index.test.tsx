import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders } from "../../../tests/test-utils";

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

describe("SpendLogsTable", () => {
  it("renders the four log tabs", () => {
    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    for (const label of ["Request Logs", "Audit Logs", "Deleted Keys", "Deleted Teams"]) {
      expect(screen.getByRole("tab", { name: label })).toBeInTheDocument();
    }
  });

  it("marks only the visible tab's panel active so background tabs do not query", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("active");

    await user.click(screen.getByRole("tab", { name: "Audit Logs" }));

    expect(await screen.findByTestId("audit-logs-panel")).toHaveTextContent("active");
    expect(screen.getByTestId("request-logs-panel")).toHaveTextContent("inactive");
  });

  describe("auth-not-ready guard", () => {
    it("shows a loading spinner when credentials are not yet resolved", () => {
      renderWithProviders(<SpendLogsTable {...defaultProps} accessToken={null} />);

      expect(document.querySelector(".ant-spin")).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Request Logs" })).not.toBeInTheDocument();
    });

    it("renders the tabs (no spinner) once all credentials are present", () => {
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      expect(document.querySelector(".ant-spin")).not.toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Request Logs" })).toBeInTheDocument();
    });
  });
});
