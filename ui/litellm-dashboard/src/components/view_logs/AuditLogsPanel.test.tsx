import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import moment from "moment";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import AuditLogsPanel from "./AuditLogsPanel";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

const uiAuditLogsCall = vi.fn();

vi.mock("../networking", () => ({
  uiAuditLogsCall: (...args: unknown[]) => uiAuditLogsCall(...args),
  serverRootPath: "",
}));

const EMPTY_RESPONSE = { audit_logs: [], total: 0, page: 1, page_size: 50, total_pages: 0 };

function renderPanel() {
  return renderWithProviders(
    <AuditLogsPanel
      accessToken="test-access-token"
      token="test-token"
      userRole="Admin"
      userID="user-1"
      isActive={true}
      premiumUser={true}
    />,
  );
}

async function applyFilter(fill: () => Promise<void> | void) {
  const user = userEvent.setup();
  await user.click(screen.getByTestId("datatable-filters-trigger"));
  await fill();
  await user.click(screen.getByTestId("filter-drawer-apply"));
}

function lastCallParams(): Record<string, unknown> {
  const lastCall = uiAuditLogsCall.mock.calls.at(-1)?.[0] as { params: Record<string, unknown> };
  return lastCall.params;
}

describe("AuditLogsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    uiAuditLogsCall.mockResolvedValue(EMPTY_RESPONSE);
  });

  it("sends the team filter as the object_team query param", async () => {
    renderPanel();
    await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

    const user = userEvent.setup();
    await applyFilter(async () => {
      await user.type(await screen.findByPlaceholderText("Team ID or alias…"), "ml-platform");
    });

    await waitFor(() => {
      expect(lastCallParams().object_team).toBe("ml-platform");
    });
    expect(lastCallParams().object_team_id).toBeUndefined();
  });

  it("sends the date range as UTC start_date and end_date query params", async () => {
    renderPanel();
    await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

    await applyFilter(async () => {
      fireEvent.change(await screen.findByTestId("audit-filter-start-date"), {
        target: { value: "2026-07-01T10:00" },
      });
      fireEvent.change(screen.getByTestId("audit-filter-end-date"), { target: { value: "2026-07-02T18:30" } });
    });

    await waitFor(() => {
      expect(lastCallParams().start_date).toBe(moment("2026-07-01T10:00").utc().format("YYYY-MM-DD HH:mm:ss"));
    });
    expect(lastCallParams().end_date).toBe(moment("2026-07-02T18:30").utc().format("YYYY-MM-DD HH:mm:ss"));
  });

  it("keeps the pre-existing filters mapped to their unchanged query params", async () => {
    renderPanel();
    await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

    const user = userEvent.setup();
    await applyFilter(async () => {
      await user.type(await screen.findByPlaceholderText("Enter object ID…"), "obj-1");
      await user.type(screen.getByPlaceholderText("Enter user ID…"), "changer-1");
      await user.type(screen.getByPlaceholderText("Enter key hash…"), "hash-1");
    });

    await waitFor(() => {
      expect(lastCallParams().object_id).toBe("obj-1");
    });
    expect(lastCallParams().changed_by).toBe("changer-1");
    expect(lastCallParams().object_key_hash).toBe("hash-1");
  });
});
