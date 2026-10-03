import type { Job, Lens, Review } from "../model/types";

// TODO: remove once workers in every environment report reviews; it only fills jobs whose reviews list is empty
const MODEL = "cerebras/gpt-oss-120b";

export const fixtureReviews: readonly Review[] = [
  {
    execution_id: "fixture-a91f3c02",
    trace_id: "a91f3c02",
    agent: "support-bot",
    name: "order status",
    model: MODEL,
    duration_ms: 410,
    at: "2026-10-03T16:00:01Z",
    cannot_assess: false,
    reasoning:
      "The user asked for the status of order A-2291. lookup_order failed with a 500, so no order data came back. The reply still gives a ship date that appears in no tool output, so the agent made the answer up.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "Where is my order #A-2291?", cited: false },
      { span_id: "s2", kind: "tool", name: "lookup_order", preview: 'lookup_order(order_id="A-2291")', cited: false },
      { span_id: "s3", kind: "tool", name: "lookup_order", preview: "500 Internal Server Error", cited: true },
      {
        span_id: "s4",
        kind: "llm",
        name: "reply",
        preview: "Your order shipped yesterday and arrives Friday.",
        cited: true,
      },
    ],
    verdicts: [{ check_id: "invented", kind: "issue", summary: "Invented an order status after a failed lookup" }],
  },
  {
    execution_id: "fixture-5c0e7b19",
    trace_id: "5c0e7b19",
    agent: "support-bot",
    name: "returns",
    model: MODEL,
    duration_ms: 350,
    at: "2026-10-03T16:00:02Z",
    cannot_assess: false,
    reasoning:
      "The user asked about a return 12 days after purchase. get_policy returned a 30-day window and the reply matches it, so the answer is grounded.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "Can I return my headphones? Bought 12 days ago.", cited: false },
      { span_id: "s2", kind: "tool", name: "get_policy", preview: "Returns accepted within 30 days.", cited: false },
      {
        span_id: "s3",
        kind: "llm",
        name: "reply",
        preview: "Yes, you're within the 30-day window. Here's the label.",
        cited: false,
      },
    ],
    verdicts: [],
  },
  {
    execution_id: "fixture-e2d48a6f",
    trace_id: "e2d48a6f",
    agent: "support-bot",
    name: "cancel subscription",
    model: MODEL,
    duration_ms: 520,
    at: "2026-10-03T16:00:03Z",
    cannot_assess: false,
    reasoning:
      "The user asked to cancel three times. The agent never called cancel_subscription and kept offering billing help, so the user had to repeat themselves and the task never got done.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "cancel my subscription", cited: false },
      { span_id: "s2", kind: "llm", name: "reply", preview: "I can help with billing questions!", cited: false },
      { span_id: "s3", kind: "user", name: "input", preview: "no. CANCEL my subscription", cited: true },
      { span_id: "s4", kind: "llm", name: "reply", preview: "Sure, what billing question do you have?", cited: true },
      { span_id: "s5", kind: "user", name: "input", preview: "cancel. it.", cited: false },
    ],
    verdicts: [{ check_id: "unhappy", kind: "issue", summary: "User had to repeat a cancel request 3 times" }],
  },
  {
    execution_id: "fixture-7d21aa90",
    trace_id: "7d21aa90",
    agent: "billing-agent",
    name: "refund",
    model: MODEL,
    duration_ms: 480,
    at: "2026-10-03T16:00:04Z",
    cannot_assess: false,
    reasoning:
      "The user asked for a refund on invoice 8812. issue_refund returned a timeout, but the agent told the user the refund was processed, so it ignored the failed call.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "Please refund invoice 8812", cited: false },
      { span_id: "s2", kind: "tool", name: "issue_refund", preview: "TimeoutError: upstream did not respond", cited: true },
      { span_id: "s3", kind: "llm", name: "reply", preview: "Done! Your refund has been processed.", cited: true },
    ],
    verdicts: [{ check_id: "invented", kind: "issue", summary: "Claimed a refund succeeded after a timeout" }],
  },
];

export function previewJob(lens: Lens): Job {
  const now = new Date().toISOString();
  return {
    id: `${lens.id}-preview`,
    status: "running",
    stage: "Reading executions",
    trigger: "manual",
    created_at: now,
    start: now,
    end: now,
    revision: 0,
    attempts: 0,
    error: "",
    cost: 0,
    settings: lens.settings,
    assessments: [],
    steps: [],
    reviews: [],
    reviewed: 0,
    coverage: {
      eligible: fixtureReviews.length,
      selected: fixtureReviews.length,
      screened: 0,
      investigated: 0,
      inconclusive: 0,
      grouping_batches: 0,
      grouped_batches: 0,
      candidates: 0,
      partial: 0,
      unassessable: 0,
    },
  };
}

export function sampleBatch(tick: number, size: number): Review[] {
  return Array.from({ length: size }, (_, offset) => {
    const index = tick * size + offset;
    const base = fixtureReviews[index % fixtureReviews.length];
    return {
      ...base,
      execution_id: `${base.execution_id}-${index}`,
      at: new Date(Date.parse(base.at) + index * 1000).toISOString(),
    };
  });
}
