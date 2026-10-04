import type { Review } from "../model/types";

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
      {
        span_id: "s1",
        kind: "user",
        name: "input",
        preview: "Can I return my headphones? Bought 12 days ago.",
        cited: false,
      },
      { span_id: "s2", kind: "tool", name: "get_policy", preview: "Returns accepted within 30 days.", cited: false },
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
      "The user asked to cancel three times. The agent never called cancel_subscription, so the user had to repeat themselves.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "cancel my subscription", cited: false },
      { span_id: "s2", kind: "user", name: "input", preview: "no. CANCEL my subscription", cited: true },
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
    reasoning: "issue_refund timed out, but the agent told the user the refund was processed.",
    spans: [
      { span_id: "s1", kind: "user", name: "input", preview: "Please refund invoice 8812", cited: false },
      {
        span_id: "s2",
        kind: "tool",
        name: "issue_refund",
        preview: "TimeoutError: upstream did not respond",
        cited: true,
      },
    ],
    verdicts: [{ check_id: "invented", kind: "issue", summary: "Claimed a refund succeeded after a timeout" }],
  },
];
