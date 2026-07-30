import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import UsagePage from "./usage";

const networking = vi.hoisted(() => ({
  adminSpendLogsCall: vi.fn(),
  adminTopKeysCall: vi.fn(),
  adminTopModelsCall: vi.fn(),
  adminTopEndUsersCall: vi.fn(),
  teamSpendLogsCall: vi.fn(),
  tagsSpendLogsCall: vi.fn(),
  allTagNamesCall: vi.fn(),
  adminspendByProvider: vi.fn(),
  adminGlobalActivity: vi.fn(),
  adminGlobalActivityPerModel: vi.fn(),
  getProxyUISettings: vi.fn(),
  modelAvailableCall: vi.fn(),
  keyInfoV1Call: vi.fn(),
}));

vi.mock("@/components/networking", () => networking);
vi.mock("../../../../components/networking", () => networking);

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "sk-test",
    token: "tok",
    userRole: "Admin",
    userId: "u1",
    premiumUser: true,
  }),
}));

const UNLIMITED_SETTINGS = { DISABLE_EXPENSIVE_DB_QUERIES: false, NUM_SPEND_LOGS_ROWS: 10 };

const renderUsage = (overrides: Partial<React.ComponentProps<typeof UsagePage>> = {}) =>
  renderWithProviders(
    <UsagePage
      accessToken="sk-test"
      token="tok"
      userRole="Admin"
      userID="u1"
      keys={null}
      premiumUser={true}
      {...overrides}
    />,
  );

beforeEach(() => {
  vi.clearAllMocks();
  networking.getProxyUISettings.mockResolvedValue(UNLIMITED_SETTINGS);
  networking.adminSpendLogsCall.mockResolvedValue([{ date: "2026-07-01", spend: 12.5 }]);
  networking.adminTopKeysCall.mockResolvedValue([
    { api_key: "sk-abcdefghijk", key_alias: "prod-key", total_spend: 9.5 },
  ]);
  networking.adminTopModelsCall.mockResolvedValue([{ model: "gpt-5.1", total_spend: 7.25 }]);
  networking.adminTopEndUsersCall.mockResolvedValue([
    { end_user: "customer-alpha", total_spend: 3.5, total_count: 42 },
  ]);
  networking.teamSpendLogsCall.mockResolvedValue({
    daily_spend: [{ date: "2026-07-01", "team-a": 5 }],
    teams: ["team-a"],
    total_spend_per_team: [{ team_id: "team-a", total_spend: 5 }],
  });
  networking.tagsSpendLogsCall.mockResolvedValue({ spend_per_tag: [{ name: "prod", spend: 4 }] });
  networking.allTagNamesCall.mockResolvedValue({ tag_names: ["prod", "staging"] });
  networking.adminspendByProvider.mockResolvedValue([{ provider: "openai", spend: 6.75 }]);
  networking.adminGlobalActivity.mockResolvedValue({
    sum_api_requests: 120,
    sum_total_tokens: 4500,
    daily_data: [{ date: "2026-07-01", api_requests: 120, total_tokens: 4500 }],
  });
  networking.adminGlobalActivityPerModel.mockResolvedValue([]);
  networking.modelAvailableCall.mockResolvedValue({ data: [] });
  networking.keyInfoV1Call.mockResolvedValue({ info: {} });
});

describe("old usage page", () => {
  describe("when the proxy has disabled expensive DB queries", () => {
    beforeEach(() => {
      networking.getProxyUISettings.mockResolvedValue({
        DISABLE_EXPENSIVE_DB_QUERIES: true,
        NUM_SPEND_LOGS_ROWS: 2500000,
      });
    });

    it("shows the database query limit warning instead of the usage dashboard", async () => {
      renderUsage();

      expect(await screen.findByText("Database Query Limit Reached")).toBeInTheDocument();
      expect(screen.getByText(/SpendLogs in DB has/)).toHaveTextContent("2500000");
      expect(screen.getByText(/Please follow our guide to view usage when SpendLogs has more than 1M rows/i));
      expect(screen.queryByRole("tab", { name: "All Up" })).not.toBeInTheDocument();
    });

    it("links to the cost tracking guide in a new tab", async () => {
      renderUsage();

      const link = await screen.findByRole("link", { name: "View Usage Guide" });
      expect(link).toHaveAttribute("href", "https://docs.litellm.ai/docs/proxy/cost_tracking");
      expect(link).toHaveAttribute("target", "_blank");
    });

    it("skips every expensive usage query", async () => {
      renderUsage();

      await screen.findByText("Database Query Limit Reached");
      await waitFor(() => expect(networking.getProxyUISettings).toHaveBeenCalled());

      expect(networking.adminSpendLogsCall).not.toHaveBeenCalled();
      expect(networking.adminspendByProvider).not.toHaveBeenCalled();
      expect(networking.adminTopKeysCall).not.toHaveBeenCalled();
      expect(networking.adminTopModelsCall).not.toHaveBeenCalled();
      expect(networking.adminGlobalActivity).not.toHaveBeenCalled();
      expect(networking.adminGlobalActivityPerModel).not.toHaveBeenCalled();
      expect(networking.teamSpendLogsCall).not.toHaveBeenCalled();
      expect(networking.adminTopEndUsersCall).not.toHaveBeenCalled();
      expect(networking.tagsSpendLogsCall).not.toHaveBeenCalled();
    });
  });

  describe("as an admin", () => {
    it("renders the admin tabs", async () => {
      renderUsage();

      expect(await screen.findByRole("tab", { name: "All Up" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Team Based Usage" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Customer Usage" })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Tag Based Usage" })).toBeInTheDocument();
    });

    it("renders the cost panel cards", async () => {
      renderUsage();

      expect(await screen.findByText("Monthly Spend")).toBeInTheDocument();
      expect(screen.getByText("Top Virtual Keys")).toBeInTheDocument();
      expect(screen.getByText("Top Models")).toBeInTheDocument();
      expect(screen.getByText("Spend by Provider")).toBeInTheDocument();
    });

    it("lists spend by provider in a table", async () => {
      renderUsage();

      const providerCell = await screen.findByText("openai");
      const row = providerCell.closest("tr");
      expect(row).not.toBeNull();
      expect(within(row as HTMLElement).getByText("$6.75")).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "Provider" })).toBeInTheDocument();
    });

    it("shows the customer usage table when its tab is selected", async () => {
      const user = userEvent.setup();
      renderUsage();

      await user.click(await screen.findByRole("tab", { name: "Customer Usage" }));

      const customerCell = await screen.findByText("customer-alpha");
      const row = customerCell.closest("tr");
      expect(row).not.toBeNull();
      expect(within(row as HTMLElement).getByText("$3.50")).toBeInTheDocument();
      expect(within(row as HTMLElement).getByText("42")).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "Total Events" })).toBeInTheDocument();
    });

    it("shows the tag spend panel when its tab is selected", async () => {
      const user = userEvent.setup();
      renderUsage();

      await user.click(await screen.findByRole("tab", { name: "Tag Based Usage" }));

      expect(await screen.findByText("Spend Per Tag")).toBeInTheDocument();
    });

    it("shows the team spend panel when its tab is selected", async () => {
      const user = userEvent.setup();
      renderUsage();

      await user.click(await screen.findByRole("tab", { name: "Team Based Usage" }));

      expect(await screen.findByText("Total Spend Per Team")).toBeInTheDocument();
      expect(screen.getByText("Daily Spend Per Team")).toBeInTheDocument();
    });
  });

  describe("as a non-admin", () => {
    it("renders only the All Up tab and skips admin-only queries", async () => {
      renderUsage({ userRole: "Internal User" });

      expect(await screen.findByRole("tab", { name: "All Up" })).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Team Based Usage" })).not.toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Customer Usage" })).not.toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Tag Based Usage" })).not.toBeInTheDocument();

      await waitFor(() => expect(networking.adminSpendLogsCall).toHaveBeenCalled());
      expect(networking.teamSpendLogsCall).not.toHaveBeenCalled();
      expect(networking.adminTopEndUsersCall).not.toHaveBeenCalled();
    });
  });
});
