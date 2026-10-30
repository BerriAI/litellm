export interface Category {
  id: string;
  name: string;
  description: string;
}
export interface Settings {
  version: string;
  phase: "sampling" | "review" | "active" | "error";
  categories: Category[];
  sample_size: number;
  sampled: number;
  error: string | null;
  discovery_model: string;
  classifier_model: string;
  team_id: string;
}
export interface Status {
  configured: boolean;
  settings: Settings | null;
}
export interface SpendBucket {
  day: string;
  category_id: string;
  model: string;
  spend: number;
  requests: number;
  unknown_cost_requests: number;
}
export interface Report {
  version: string;
  categories: Category[];
  buckets: SpendBucket[];
}
export interface Example {
  request_id: string;
  model: string;
  spend: number | null;
  excerpt: string;
}
export interface ContextFile {
  name: string;
  text: string;
}

export const money = (value: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 6 }).format(value);

export function categorySummary(report: Report) {
  const categories = [
    ...report.categories,
    {
      id: "unclassified",
      name: "Unclassified",
      description: "Pending classification, unavailable content, failed classification, or no matching category.",
    },
  ];
  return categories
    .map((category) => {
      const buckets = report.buckets.filter((bucket) => bucket.category_id === category.id);
      const models = [...new Set(buckets.map((bucket) => bucket.model))]
        .map((model) => ({
          model,
          spend: buckets.filter((bucket) => bucket.model === model).reduce((total, bucket) => total + bucket.spend, 0),
        }))
        .sort((a, b) => b.spend - a.spend);
      return {
        ...category,
        spend: buckets.reduce((total, bucket) => total + bucket.spend, 0),
        requests: buckets.reduce((total, bucket) => total + bucket.requests, 0),
        models,
      };
    })
    .sort((a, b) => b.spend - a.spend);
}

export function categoryChart(report: Report) {
  const days = [...new Set(report.buckets.map((bucket) => bucket.day))].sort();
  const categories = [...report.categories.map((category) => category.id), "unclassified"];
  return days.map((day) =>
    Object.fromEntries([
      ["day", day],
      ...categories.map((category) => [
        category,
        report.buckets
          .filter((bucket) => bucket.day === day && bucket.category_id === category)
          .reduce((sum, bucket) => sum + bucket.spend, 0),
      ]),
    ]),
  );
}

export function requestQuote(excerpt: string): string {
  try {
    const messages: unknown = JSON.parse(excerpt);
    if (!Array.isArray(messages)) return excerpt;
    const request = messages.findLast((message: unknown) => {
      if (typeof message !== "object" || message === null) return false;
      return "role" in message && message.role === "user";
    });
    if (request && typeof request.content === "string") return request.content;
    return excerpt;
  } catch {
    return excerpt;
  }
}

export function roiTotals(report: Report | undefined) {
  const rows = report ? categorySummary(report) : [];
  const total = rows.reduce((sum, row) => sum + row.spend, 0);
  const requests = rows.reduce((sum, row) => sum + row.requests, 0);
  const categorized = rows.filter((row) => row.id !== "unclassified").reduce((sum, row) => sum + row.requests, 0);
  const unknownCosts = report?.buckets.reduce((sum, row) => sum + row.unknown_cost_requests, 0) ?? 0;
  const coverage = requests ? ((100 * categorized) / requests).toFixed(1) : "0";
  const unclassifiedSpend = rows.find((row) => row.id === "unclassified")?.spend ?? 0;
  return { rows, total, requests, categorized, unknownCosts, coverage, unclassifiedSpend };
}
