import { describe, expect, it } from "vitest";
import {
  entityRows,
  errorRate,
  errorSeries,
  rateByStatusRows,
  errorSummary,
  formatRate,
  statusCodeLabel,
  statusCodeRows,
  type RequestErrorActivity,
} from "./errorsData";

const activity = (overrides: Partial<RequestErrorActivity> = {}): RequestErrorActivity => ({
  total_successful_requests: 900,
  total_failed_requests: 100,
  by_date: [
    {
      date: "2026-10-08",
      successful_requests: 400,
      failed_requests: 60,
      client_errors: 50,
      server_errors: 5,
      by_status_code: [
        { status_code: 429, failed_requests: 40 },
        { status_code: 401, failed_requests: 10 },
        { status_code: 503, failed_requests: 5 },
      ],
    },
    {
      date: "2026-10-07",
      successful_requests: 500,
      failed_requests: 40,
      client_errors: 40,
      server_errors: 0,
      by_status_code: [
        { status_code: 429, failed_requests: 30 },
        { status_code: 401, failed_requests: 10 },
      ],
    },
  ],
  by_status_code: [
    { status_code: 429, failed_requests: 70 },
    { status_code: 503, failed_requests: 5 },
    { status_code: 401, failed_requests: 20 },
  ],
  by_key: [
    {
      id: "hash-1",
      label: "prod-key",
      api_requests: 500,
      failed_requests: 60,
      top_status_code: 429,
      top_status_code_requests: 55,
    },
    {
      id: "hash-2",
      label: null,
      api_requests: 0,
      failed_requests: 0,
      top_status_code: null,
      top_status_code_requests: 0,
    },
  ],
  by_team: [],
  by_user: [],
  by_model: [
    {
      id: "gpt-4o",
      label: null,
      api_requests: 1000,
      failed_requests: 1,
      top_status_code: 0,
      top_status_code_requests: 1,
    },
  ],
  ...overrides,
});

describe("errorSummary", () => {
  it("derives the rate and the status-class callouts from the edge totals", () => {
    const expected = {
      requests: 1000,
      failed: 100,
      rate: 10,
      clientErrors: 90,
      serverErrors: 5,
      rateLimited: 70,
      authFailures: 20,
    };
    expect(errorSummary(activity())).toEqual(expected);
  });

  it("has no rate when nothing was served", () => {
    const empty = activity({ total_successful_requests: 0, total_failed_requests: 0, by_status_code: [] });
    expect(errorSummary(empty).rate).toBeNull();
  });
});

describe("errorSeries", () => {
  it("orders days and stacks the top status codes, with the unrecorded remainder as Other", () => {
    const series = errorSeries(activity());
    expect(series.codes).toEqual(["429", "401", "503", "Other"]);
    expect(series.rows).toEqual([
      {
        date: "2026-10-07",
        requests: 540,
        failed: 40,
        rate: expect.closeTo(7.41, 2),
        "429": 30,
        "401": 10,
        "503": 0,
        Other: 0,
      },
      {
        date: "2026-10-08",
        requests: 460,
        failed: 60,
        rate: expect.closeTo(13.04, 2),
        "429": 40,
        "401": 10,
        "503": 5,
        Other: 5,
      },
    ]);
  });

  it("has no Other column when every failure has a recorded status", () => {
    const complete = activity({
      by_date: [
        {
          date: "2026-10-08",
          successful_requests: 10,
          failed_requests: 2,
          client_errors: 2,
          server_errors: 0,
          by_status_code: [{ status_code: 429, failed_requests: 2 }],
        },
      ],
      by_status_code: [{ status_code: 429, failed_requests: 2 }],
    });
    expect(errorSeries(complete).codes).toEqual(["429"]);
  });
});

describe("rateByStatusRows", () => {
  it("turns each status code count into a share of that day's requests", () => {
    const rows = rateByStatusRows(errorSeries(activity()));
    expect(rows).toEqual([
      {
        date: "2026-10-07",
        requests: 540,
        failed: 40,
        rate: expect.closeTo(7.41, 2),
        "429": expect.closeTo(5.56, 2),
        "401": expect.closeTo(1.85, 2),
        "503": 0,
        Other: 0,
      },
      {
        date: "2026-10-08",
        requests: 460,
        failed: 60,
        rate: expect.closeTo(13.04, 2),
        "429": expect.closeTo(8.7, 2),
        "401": expect.closeTo(2.17, 2),
        "503": expect.closeTo(1.09, 2),
        Other: expect.closeTo(1.09, 2),
      },
    ]);
  });
});

describe("statusCodeRows", () => {
  it("ranks codes by count, labels them and adds the unrecorded remainder", () => {
    expect(statusCodeRows(activity())).toEqual([
      { status_code: 429, label: "Rate Limited", failed_requests: 70, share: 70 },
      { status_code: 401, label: "Unauthorized", failed_requests: 20, share: 20 },
      { status_code: 0, label: "Not recorded", failed_requests: 5, share: 5 },
      { status_code: 503, label: "Service Unavailable", failed_requests: 5, share: 5 },
    ]);
  });

  it("is empty when nothing failed", () => {
    expect(statusCodeRows(activity({ total_failed_requests: 0, by_status_code: [] }))).toEqual([]);
  });
});

describe("entityRows", () => {
  it("names rows by alias, computes the rate and describes the top status", () => {
    expect(entityRows(activity(), "key")).toEqual([
      { id: "hash-1", name: "prod-key", requests: 500, failed: 60, rate: 12, topStatus: "429 · 55" },
      { id: "hash-2", name: "hash-2", requests: 0, failed: 0, rate: null, topStatus: null },
    ]);
  });

  it("switches source by kind and spells out an unrecorded top status", () => {
    expect(entityRows(activity(), "team")).toEqual([]);
    expect(entityRows(activity(), "model")[0].topStatus).toBe("No status · 1");
  });
});

describe("rates", () => {
  it.each([
    [0, 0, null],
    [1, 1000, 0.1],
    [50, 200, 25],
  ])("errorRate(%s, %s)", (failed, total, expected) => {
    expect(errorRate(failed, total)).toBe(expected);
  });

  it.each([
    [null, "—"],
    [0, "0.0%"],
    [0.04, "<0.1%"],
    [12.345, "12.3%"],
  ])("formatRate(%s)", (rate, expected) => {
    expect(formatRate(rate)).toBe(expected);
  });
});

describe("statusCodeLabel", () => {
  it.each([
    [0, "Not recorded"],
    [401, "Unauthorized"],
    [429, "Rate Limited"],
    [503, "Service Unavailable"],
    [418, "Client Error"],
    [599, "Server Error"],
  ] as const)("labels status %s", (code, label) => {
    expect(statusCodeLabel(code)).toBe(label);
  });
});
