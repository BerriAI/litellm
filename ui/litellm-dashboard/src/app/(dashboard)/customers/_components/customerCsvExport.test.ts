import Papa from "papaparse";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

import { buildCustomersCsv, buildCustomersCsvRows, exportCustomersToCsv } from "./customerCsvExport";

const customer = (overrides: Partial<EndUser> = {}): EndUser => ({
  user_id: "customer-1",
  blocked: false,
  spend: 0,
  ...overrides,
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("buildCustomersCsvRows", () => {
  it("exports customer identity, status, budget, model access, and MCP grants", () => {
    const customerFields: Partial<EndUser> = {
      user_id: "customer-42",
      alias: "finance",
      blocked: true,
      spend: 12.5,
      budget_id: "budget-42",
      litellm_budget_table: {
        created_at: "2026-01-01T00:00:00Z",
        max_budget: 250,
        soft_budget: 200,
        budget_duration: "30d",
        budget_reset_at: "2026-02-01T00:00:00Z",
        tpm_limit: 1000,
        rpm_limit: 50,
        tpd_limit: 20000,
        max_parallel_requests: 10,
      },
      models: ["model-1", "model-2"],
      allowed_model_region: "eu",
      default_model: "model-1",
      object_permission: {
        object_permission_id: "permission-42",
        models: [],
        mcp_servers: ["server-1", "server-2"],
        mcp_access_groups: ["group-1"],
        mcp_toolsets: ["toolset-1"],
        agents: [],
        agent_access_groups: [],
        blocked_tools: [],
        search_tools: [],
        vector_stores: [],
      },
    };
    const [row] = buildCustomersCsvRows([customer(customerFields)]);

    const expectedRow = {
      "Customer Alias": "finance",
      "Customer ID": "customer-42",
      Status: "Blocked",
      "Spend (USD)": 12.5,
      "Budget ID": "budget-42",
      "Max Budget (USD)": 250,
      "Soft Budget (USD)": 200,
      "Budget Duration": "30d",
      "Budget Reset At": "2026-02-01T00:00:00Z",
      "TPM Limit": 1000,
      "RPM Limit": 50,
      "TPD Limit": 20000,
      "Max Parallel Requests": 10,
      Models: "model-1, model-2",
      "MCP Servers": "server-1, server-2",
      "MCP Access Groups": "group-1",
      "MCP Toolsets": "toolset-1",
      "Allowed Model Region": "eu",
      "Default Model": "model-1",
    };
    expect(row).toEqual(expectedRow);
  });

  it("keeps zero limits and spend while rendering missing optional values as empty cells", () => {
    const missingFields: Partial<EndUser> = {
      alias: null,
      budget_id: null,
      litellm_budget_table: null,
      object_permission: null,
    };
    const [minimal, zeroBudget] = buildCustomersCsvRows([
      customer(missingFields),
      customer({
        user_id: "zero-budget",
        litellm_budget_table: {
          created_at: "2026-01-01T00:00:00Z",
          budget_id: "nested-budget",
          max_budget: 0,
          tpm_limit: 0,
          rpm_limit: 0,
        },
      }),
    ]);

    expect(minimal.Status).toBe("Active");
    expect(minimal["Spend (USD)"]).toBe(0);
    expect(Object.entries(minimal).filter(([key]) => !["Customer ID", "Status", "Spend (USD)"].includes(key))).toEqual(
      Object.keys(minimal)
        .filter((key) => !["Customer ID", "Status", "Spend (USD)"].includes(key))
        .map((key) => [key, ""]),
    );
    expect(zeroBudget["Budget ID"]).toBe("nested-budget");
    expect(zeroBudget["Max Budget (USD)"]).toBe(0);
    expect(zeroBudget["TPM Limit"]).toBe(0);
    expect(zeroBudget["RPM Limit"]).toBe(0);
  });
});

describe("buildCustomersCsv", () => {
  it("round trips quotes, commas, and line breaks without splitting customer rows", () => {
    const csv = buildCustomersCsv([
      customer({ alias: 'sales, "east"\nregion', models: ["model-1", "model-2"] }),
      customer({ user_id: "customer-2" }),
    ]);
    const parsed = Papa.parse<Record<string, string>>(csv, { header: true });

    expect(parsed.errors).toEqual([]);
    expect(parsed.data).toHaveLength(2);
    expect(parsed.data[0]["Customer Alias"]).toBe('sales, "east"\nregion');
    expect(parsed.data[0].Models).toBe("model-1, model-2");
    expect(parsed.data[1]["Customer ID"]).toBe("customer-2");
  });

  it.each(["=SUM(A1:A9)", "+cmd", "-cmd", "@cmd", "\tcmd", "\rcmd", "=SUM(A1:A9)\nsecond line", "+cmd\r\nsecond line"])(
    "neutralizes formula-leading text %j",
    (alias) => {
      const csv = buildCustomersCsv([customer({ alias })]);
      const parsed = Papa.parse<Record<string, string>>(csv, { header: true });

      expect(parsed.data[0]["Customer Alias"]).toBe(`'${alias}`);
    },
  );
});

describe("exportCustomersToCsv", () => {
  it("downloads the supplied customers, returns their count, and cleans up the download URL", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-15T12:00:00Z"));
    const a = { href: "", download: "", click: vi.fn() };
    const createObjectURL = vi.fn().mockReturnValue("blob:customers-export");
    const revokeObjectURL = vi.fn();
    const createElement = vi.fn().mockReturnValue(a);
    const appendChild = vi.fn();
    const removeChild = vi.fn();
    vi.stubGlobal("window", { URL: { createObjectURL, revokeObjectURL } });
    vi.stubGlobal("document", { createElement, body: { appendChild, removeChild } });

    expect(exportCustomersToCsv([customer({ user_id: "visible-1" }), customer({ user_id: "visible-2" })])).toBe(2);

    const [blob] = createObjectURL.mock.calls[0] as [Blob];
    expect(blob.type).toBe("text/csv;charset=utf-8;");
    expect(
      Papa.parse<Record<string, string>>(await blob.text(), { header: true }).data.map((row) => row["Customer ID"]),
    ).toEqual(["visible-1", "visible-2"]);
    expect(createElement).toHaveBeenCalledWith("a");
    expect(a.href).toBe("blob:customers-export");
    expect(a.download).toBe("customers_export_2026-01-15.csv");
    expect(appendChild).toHaveBeenCalledWith(a);
    expect(a.click).toHaveBeenCalledOnce();
    expect(removeChild).toHaveBeenCalledWith(a);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:customers-export");
  });
});
