import { describe, expect, test } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { CustomerUsage } from "../customers/_components/customerPortalData";
import { rowsByCustomer, rowsByCustomerAndKey, rowsByCustomerAndModel, sumRows, toCsv } from "./chargebackRows";

const totals = (spend: number, requests = 1) => ({
  spend,
  requests,
  inputTokens: requests * 10,
  outputTokens: requests * 5,
});

const usage: ReadonlyMap<string, CustomerUsage> = new Map([
  [
    "acme",
    {
      ...totals(6, 3),
      keys: [
        { hash: "sk-hash-aaaaaaaaaaaa", alias: "prod", ...totals(5, 2) },
        { hash: "sk-hash-bbbbbbbbbbbb", alias: null, ...totals(1, 1) },
      ],
    },
  ],
  ["idle", { ...totals(0, 0), keys: [] }],
  ["globex", { ...totals(9, 1), keys: [{ hash: "h3", alias: "main", ...totals(9, 1) }] }],
]);

describe("rowsByCustomer", () => {
  test("one row per customer with traffic, highest spend first", () => {
    expect(rowsByCustomer(usage).map((r) => [r.customer, r.dimension, r.spend])).toEqual([
      ["globex", null, 9],
      ["acme", null, 6],
    ]);
  });
});

describe("rowsByCustomerAndKey", () => {
  test("one row per customer and key, labelled by alias or hash prefix", () => {
    expect(rowsByCustomerAndKey(usage).map((r) => [r.customer, r.dimension, r.spend])).toEqual([
      ["globex", "main", 9],
      ["acme", "prod", 5],
      ["acme", "sk-hash-bbbb", 1],
    ]);
  });
});

describe("rowsByCustomerAndModel", () => {
  const metrics = (spend: number): SpendMetrics => ({
    spend,
    prompt_tokens: 1,
    completion_tokens: 1,
    total_tokens: 2,
    api_requests: 1,
    successful_requests: 1,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  });
  const day = (models: Record<string, number>): DailyData => ({
    date: "2026-11-01",
    metrics: metrics(0),
    breakdown: {
      models: Object.fromEntries(
        Object.entries(models).map(([m, s]) => [m, { metrics: metrics(s), metadata: {}, api_key_breakdown: {} }]),
      ),
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
    },
  });
  test("sums each customer's models across days", () => {
    const rows = rowsByCustomerAndModel(new Map([["acme", [day({ "gpt-5": 1, claude: 2 }), day({ "gpt-5": 4 })]]]));
    expect(rows.map((r) => [r.customer, r.dimension, r.spend, r.requests])).toEqual([
      ["acme", "gpt-5", 5, 2],
      ["acme", "claude", 2, 1],
    ]);
  });
});

describe("sumRows and toCsv", () => {
  test("total row matches the sum of the rows", () => {
    const expected = { spend: 15, requests: 4, inputTokens: 40, outputTokens: 20 };
    expect(sumRows(rowsByCustomer(usage))).toEqual(expected);
  });
  test("csv carries alias, optional dimension column and escaped quotes", () => {
    const rows = rowsByCustomerAndKey(usage).slice(0, 1);
    const csv = toCsv(rows, new Map([["globex", 'Globex "Inc"']]), "key");
    expect(csv.split("\n")).toEqual([
      '"customer_id","customer","key","requests","input_tokens","output_tokens","spend_usd"',
      '"globex","Globex ""Inc""","main","1","10","5","9.000000"',
    ]);
  });
  test("csv neutralises ids and aliases that a spreadsheet would run as a formula", () => {
    const rows = rowsByCustomer(new Map([['=HYPERLINK("x")', usage.get("acme")!]]));
    const line = toCsv(rows, new Map([['=HYPERLINK("x")', "-1+1"]]), null).split("\n")[1];
    expect(line.startsWith(`"'=HYPERLINK(""x"")","'-1+1",`)).toBe(true);
    expect(line.endsWith('"3","30","15","6.000000"')).toBe(true);
  });
  test("csv without a dimension has no dimension column", () => {
    expect(toCsv(rowsByCustomer(usage), new Map(), null).split("\n")[0]).toBe(
      '"customer_id","customer","requests","input_tokens","output_tokens","spend_usd"',
    );
  });
});
