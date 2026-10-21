export type Scenario = {
  agent: string;
  question: string;
  tool: string;
  result: string;
  answer: string;
  failed?: boolean;
};

export const scenarios: Scenario[] = [
  {
    agent: "support_agent",
    question: "Where is order #1042?",
    tool: "lookup_order",
    result: "Order service timed out",
    answer: "I will check that for you.",
    failed: true,
  },
  {
    agent: "research_agent",
    question: "Which database should we use for analytics?",
    tool: "search_docs",
    result: "Column stores are designed for analytical queries. No comparative benchmark is available.",
    answer: "The column store is 40% faster than the row store for your workload.",
  },
  {
    agent: "support_agent",
    question: "Can I return my headphones?",
    tool: "lookup_return_policy",
    result: "Returns are accepted within 30 days. This order was delivered 10 days ago.",
    answer: "Yes. Your order is eligible for a return. I can help you start it.",
  },
  {
    agent: "release_agent",
    question: "Is the checkout release ready to ship?",
    tool: "read_test_results",
    result: "Checkout: 120 passed, 0 failed. Rollback plan approved.",
    answer: "All 120 checkout tests passed and the rollback plan is approved. The release is ready.",
  },
  {
    agent: "support_agent",
    question: "My order #1088 has not arrived",
    tool: "lookup_order",
    result: "Order service timed out",
    answer: "I will check that for you.",
    failed: true,
  },
  {
    agent: "research_agent",
    question: "How long do we retain audit logs?",
    tool: "search_docs",
    result: "Audit log retention is 90 days on the current plan.",
    answer: "Audit logs are retained for 90 days, according to the plan documentation.",
  },
  {
    agent: "release_agent",
    question: "Review the search release",
    tool: "read_test_results",
    result: "Search: 117 passed, 3 failed. Cases 17, 63 and 104 returned empty results.",
    answer: "Hold the release. Three of 120 cases returned empty results. Review cases 17, 63 and 104 before shipping.",
  },
  {
    agent: "support_agent",
    question: "Update the address on order #1091",
    tool: "lookup_order",
    result: "Order #1091 has not shipped. Address changes are allowed.",
    answer: "Your order has not shipped. Please confirm the new delivery address.",
  },
  {
    agent: "research_agent",
    question: "Can this plan store data in Europe?",
    tool: "search_docs",
    result: "Regional availability is not documented for this plan.",
    answer: "I could not verify European data residency for this plan. Please confirm with support before choosing it.",
  },
  {
    agent: "support_agent",
    question: "Check the delivery date for order #1107",
    tool: "lookup_order",
    result: "Order service timed out",
    answer: "The order service is unavailable. I can connect you to support instead.",
    failed: true,
  },
  {
    agent: "release_agent",
    question: "Summarize the authentication release checks",
    tool: "read_test_results",
    result: "Authentication: 214 passed, 0 failed. Security review approved.",
    answer: "All 214 tests passed and the security review is approved.",
  },
  {
    agent: "research_agent",
    question: "Does the API support request retries?",
    tool: "search_docs",
    result: "Retry requests with exponential backoff after a 429 or 503 response.",
    answer: "Yes. Use exponential backoff for 429 and 503 responses, as documented.",
  },
];
