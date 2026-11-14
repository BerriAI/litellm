import { describe, expect, it, vi, beforeEach } from "vitest";
import { apiClient } from "@/components/networking";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import {
  customerOrganizationRows,
  customerOrganizationsCall,
  customerOrganizationsUsageCall,
  sortCustomerOrganizations,
  type CustomerOrganization,
} from "./customerOrganizations";
import { mergeCustomerActivityPages } from "./customerUsageData";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn() } }));
const get = vi.mocked(apiClient.get);
const org = (id: string, overrides: Partial<CustomerOrganization> = {}): CustomerOrganization => ({
  customer_org_id: id,
  alias: id,
  spend: 0,
  member_count: 0,
  default_budgets: [],
  budget_groups_revision: "revision",
  created_at: "2024-03-01T00:00:00Z",
  updated_at: "2024-03-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  ...overrides,
});
const activity = (id: string, spend: number): DailyActivityAggregatedResponse => {
  const metrics = {
    spend,
    api_requests: 5,
    successful_requests: 4,
    failed_requests: 1,
    prompt_tokens: 50,
    completion_tokens: 20,
    total_tokens: 70,
    cache_read_input_tokens: 10,
    cache_creation_input_tokens: 0,
    total_response_time_ms: 1000,
    timed_requests: 4,
    autorouter_savings_spend: 0,
    compression_saved_tokens: 0,
    compression_savings_spend: 0,
    flat_cost: 0,
    gateway_injected_caching_savings_spend: 0,
    guardrail_spend: 0,
    prompt_caching_savings_spend: 0,
    ptu_hours: 0,
  };
  return {
    metadata: EMPTY_DAILY_ACTIVITY_METADATA,
    results: [{ date: "2024-03-15", metrics, breakdown: { entities: { [id]: { metrics } } } }],
  };
};

beforeEach(() => vi.clearAllMocks());
const usageRequest = (ids: readonly string[]) => ({
  accessToken: "test-token",
  ids,
  startDate: "2024-03-01",
  endDate: "2024-03-31",
});

describe("complete customer organization usage", () => {
  it("loads every organization page and deduplicates identities", async () => {
    get
      .mockResolvedValueOnce({ data: [org("a")], meta: { total_pages: 2 } })
      .mockResolvedValueOnce({ data: [org("a"), org("b")], meta: { total_pages: 2 } });
    expect((await customerOrganizationsCall("test-token")).map((row) => row.customer_org_id)).toEqual(["a", "b"]);
    expect(get).toHaveBeenLastCalledWith(
      "/management/v1/customer_organizations",
      expect.objectContaining({ query: { page: 2, page_size: 100 } }),
    );
  });

  it("merges all batches without double counting organizations or daily metrics", async () => {
    const ids = Array.from({ length: 105 }, (_, index) => `org-${index}`);
    get.mockResolvedValueOnce(activity(ids[0], 60)).mockResolvedValueOnce(activity(ids[104], 40));
    const result = await customerOrganizationsUsageCall(usageRequest([...ids, ids[0]]));
    expect(get).toHaveBeenCalledTimes(2);
    const expectedTotals = {
      total_spend: 100,
      total_api_requests: 10,
      total_successful_requests: 8,
      total_failed_requests: 2,
      total_tokens: 140,
      total_cache_read_input_tokens: 20,
      total_response_time_ms: 2000,
      total_timed_requests: 8,
    };
    expect(result.metadata).toMatchObject(expectedTotals);
    expect(result.results).toHaveLength(1);
    const rows = customerOrganizationRows(
      ids.map((id) => org(id)),
      result,
    );
    expect(rows[0].share).toBe(0.6);
    expect(rows[104].share).toBe(0.4);
    expect(sortCustomerOrganizations(rows, ids[104], [{ id: "share", desc: true }])[0].share).toBe(0.4);
  });

  it("fails the aggregate when one batch fails or returns partial totals", async () => {
    get.mockResolvedValueOnce(activity("a", 60)).mockRejectedValueOnce(new Error("Unavailable"));
    await expect(
      customerOrganizationsUsageCall(usageRequest(Array.from({ length: 101 }, (_, index) => String(index)))),
    ).rejects.toThrow("Unavailable");
    get.mockResolvedValueOnce({ ...activity("a", 60), metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA, has_more: true } });
    await expect(customerOrganizationsUsageCall(usageRequest(["a"]))).rejects.toThrow();
  });

  it("includes exact IDs containing commas or surrounding spaces without losing other customers", async () => {
    const ids = ["normal", "customer,one", " spaced "];
    get
      .mockResolvedValueOnce(activity(ids[0], 60))
      .mockResolvedValueOnce(activity(ids[1], 30))
      .mockResolvedValueOnce(activity(ids[2], 10));
    const result = await customerOrganizationsUsageCall(usageRequest([...ids, ids[1]]));
    expect(result.metadata.total_spend).toBe(100);
    expect(
      customerOrganizationRows(
        ids.map((id) => org(id)),
        result,
      ).map((row) => row.share),
    ).toEqual([0.6, 0.3, 0.1]);
    expect(get.mock.calls.map(([, options]) => options?.query)).toEqual([
      { start_date: "2024-03-01", end_date: "2024-03-31", customer_org_ids: "normal" },
      { start_date: "2024-03-01", end_date: "2024-03-31", customer_org_id: "customer,one" },
      { start_date: "2024-03-01", end_date: "2024-03-31", customer_org_id: " spaced " },
    ]);
  });

  it("returns an empty aggregate without an unscoped request", async () => {
    const result = await customerOrganizationsUsageCall(usageRequest([]));
    expect(result.metadata.total_spend).toBe(0);
    expect(get).not.toHaveBeenCalled();
  });
});

describe("customer organization sorting and budgets", () => {
  it("sorts the full list before pagination with newest customers first by default", () => {
    const organizations = Array.from({ length: 30 }, (_, index) =>
      org(`org-${String(index).padStart(2, "0")}`, {
        created_at: new Date(Date.UTC(2024, 2, index + 1)).toISOString(),
      }),
    );
    const rows = customerOrganizationRows(
      organizations,
      mergeCustomerActivityPages([activity("org-00", 90), activity("org-29", 10)]),
    );
    expect(sortCustomerOrganizations(rows, "", []).slice(0, 25)[0].organization.customer_org_id).toBe("org-29");
    expect(sortCustomerOrganizations(rows, "", [{ id: "spend", desc: true }])[0].organization.customer_org_id).toBe(
      "org-00",
    );
    expect(sortCustomerOrganizations(rows, "", [{ id: "share", desc: true }])[0].share).toBe(0.9);
    expect(
      sortCustomerOrganizations(rows, "ORG-2", [{ id: "customer", desc: false }]).map(
        (row) => row.organization.customer_org_id,
      ),
    ).toEqual(organizations.slice(20).map((row) => row.customer_org_id));
  });

  it("uses recorded monthly counters for spend sorting and all budget windows for status", () => {
    const rows = customerOrganizationRows(
      [
        org("annual", {
          budget_usage: [
            { source: "scalar", budget_duration: "1mo", spend: 8, max_budget: 10, reset_at: null },
            { source: "window", budget_duration: "12mo", spend: 100, max_budget: 100, reset_at: null },
          ],
        }),
        org("unknown", { budget: { budget_limits: [{ budget_duration: "1mo", max_budget: 50 }] } }),
        org("zero", {
          budget_usage: [{ source: "scalar", budget_duration: "1mo", spend: 0, max_budget: 0, reset_at: null }],
        }),
        org("uncapped"),
      ],
      mergeCustomerActivityPages([activity("annual", 20)]),
    );
    const expectedBudget = {
      monthlySpend: 20,
      displaySpend: 8,
      share: 1,
      status: "customerPortal.analytics.reachedLimit",
    };
    expect(rows[0]).toMatchObject(expectedBudget);
    expect(rows[1]).toMatchObject({ displaySpend: null, status: "customerPortal.analytics.budgetUsageUnavailable" });
    expect(rows[2].status).toBe("customerPortal.analytics.reachedLimit");
    expect(rows[3]).toMatchObject({ share: 0, status: "customerPortal.analytics.noLimit" });
    expect(
      sortCustomerOrganizations(rows, "", [{ id: "spend", desc: false }]).at(-1)?.organization.customer_org_id,
    ).toBe("unknown");
    expect(sortCustomerOrganizations(rows, "", [{ id: "status", desc: true }])[0].status).toBe(
      "customerPortal.analytics.reachedLimit",
    );
  });
});
