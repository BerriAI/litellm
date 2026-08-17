import { waitFor } from "@testing-library/react";
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { GuardrailDetail } from "./GuardrailDetail";
import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";

// Pinned off UTC on purpose: where the local day and the UTC day coincide, a bare
// date and the resolved instants are the same request and this proves nothing.
const originalTimezone = process.env.TZ;
beforeAll(() => {
  process.env.TZ = "Asia/Kolkata";
});
afterAll(() => {
  process.env.TZ = originalTimezone;
});

const fetchMock = vi.fn();

const emptyOkResponse = {
  ok: true,
  status: 200,
  statusText: "OK",
  json: async () => ({ logs: [], total: 0, rows: [], chart: [], series: [] }),
};

const usageLogsQuery = (): URLSearchParams | undefined => {
  const call = fetchMock.mock.calls.map(([url]) => String(url)).find((url) => url.includes("/guardrails/usage/logs"));
  return call === undefined ? undefined : new URL(call, "http://x").searchParams;
};

describe("GuardrailDetail sends the viewer's local day as instants", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    fetchMock.mockResolvedValue(emptyOkResponse);
    vi.stubGlobal("fetch", fetchMock);

    renderWithProviders(
      <GuardrailDetail
        guardrailId="g-1"
        onBack={() => {}}
        accessToken="sk-test"
        startDate="2026-08-10"
        endDate="2026-08-10"
      />,
    );
  });

  it("resolves the picker's dates instead of forwarding them verbatim", async () => {
    await waitFor(() => expect(usageLogsQuery()).toBeDefined());
    const params = usageLogsQuery();

    // 00:00 and 23:59:59 on 2026-08-10 in IST, expressed as instants.
    expect(params?.get("start_date")).toBe("2026-08-09T18:30:00.000Z");
    expect(params?.get("end_date")).toBe("2026-08-10T18:29:59.999Z");
  });

  it("never sends a bare calendar date, which the endpoint would read as UTC", async () => {
    await waitFor(() => expect(usageLogsQuery()).toBeDefined());
    const params = usageLogsQuery();

    expect(params?.get("start_date")).not.toBe("2026-08-10");
    expect(params?.get("end_date")).not.toBe("2026-08-10");
  });
});
