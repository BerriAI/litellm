import { describe, expect, it } from "vitest";

import traceList from "../__fixtures__/trace_list.json";
import { fromSorting, NEWEST, orderRuns, toSorting, type RunOrder } from "./runOrder";
import type { TracePage, TraceSummary } from "../types";

const template = (traceList as TracePage).data[0] as TraceSummary;
const run = (overrides: Partial<TraceSummary>): TraceSummary => ({ ...template, ...overrides });

const RUN_OVERRIDES: readonly Partial<TraceSummary>[] = [
  {
    trace_id: "a",
    trace_ref: "ref-a",
    start_time: "2026-09-30T06:00:00Z",
    duration_ms: 500,
    span_count: 3,
    error_count: 0,
  },
  {
    trace_id: "b",
    trace_ref: "ref-b",
    start_time: "2026-09-30T07:00:00Z",
    duration_ms: 500,
    span_count: 9,
    error_count: 2,
  },
  {
    trace_id: "c",
    trace_ref: "ref-c",
    start_time: "2026-09-30T05:00:00Z",
    duration_ms: 50,
    span_count: 1,
    error_count: 1,
  },
];
const RUNS: TraceSummary[] = RUN_OVERRIDES.map(run);

const ids = (runs: readonly TraceSummary[]) => runs.map((item) => item.trace_id);

describe("orderRuns", () => {
  it.each<{ order: RunOrder; expected: string[] }>([
    { order: NEWEST, expected: ["b", "a", "c"] },
    { order: { key: "start_ms", descending: false }, expected: ["c", "a", "b"] },
    { order: { key: "span_count", descending: true }, expected: ["b", "a", "c"] },
    { order: { key: "error_count", descending: false }, expected: ["a", "c", "b"] },
  ])("orders by $order.key with descending=$order.descending", ({ order, expected }) => {
    expect(ids(orderRuns(RUNS, order))).toEqual(expected);
  });

  it("breaks a tie on the trace reference in the same direction, so pages never overlap", () => {
    expect(ids(orderRuns(RUNS, { key: "duration_ms", descending: true }))).toEqual(["b", "a", "c"]);
    expect(ids(orderRuns(RUNS, { key: "duration_ms", descending: false }))).toEqual(["c", "a", "b"]);
  });

  it("leaves the input untouched", () => {
    const before = ids(RUNS);
    orderRuns(RUNS, { key: "span_count", descending: false });
    expect(ids(RUNS)).toEqual(before);
  });
});

describe("sorting state round trip", () => {
  it("maps an order to one sorting entry and back", () => {
    const order: RunOrder = { key: "duration_ms", descending: false };
    expect(toSorting(order)).toEqual([{ id: "duration_ms", desc: false }]);
    expect(fromSorting(toSorting(order), NEWEST)).toEqual(order);
  });

  it("applies a functional updater against the current order", () => {
    const flipped = fromSorting((previous) => previous.map((entry) => ({ ...entry, desc: !entry.desc })), NEWEST);
    expect(flipped).toEqual({ key: "start_ms", descending: false });
  });

  it("keeps the current order when the state is empty or names a column the server cannot sort", () => {
    expect(fromSorting([], NEWEST)).toEqual(NEWEST);
    expect(fromSorting([{ id: "cost", desc: true }], NEWEST)).toEqual(NEWEST);
  });
});
