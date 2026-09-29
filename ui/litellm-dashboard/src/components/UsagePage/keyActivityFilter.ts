import type { ModelActivityData } from "./types";

export type KeyQuery =
  | { readonly kind: "all" }
  | { readonly kind: "substring"; readonly needle: string }
  | { readonly kind: "pattern"; readonly regex: RegExp }
  | { readonly kind: "invalid"; readonly source: string };

export function parseKeyQuery(query: string): KeyQuery {
  const trimmedQuery = query.trim();
  if (trimmedQuery === "") return { kind: "all" };

  const regexLiteral = new RegExp("^/(.+)/([a-z]*)$", "s").exec(trimmedQuery);
  if (regexLiteral !== null) {
    const [, body, flags] = regexLiteral;
    try {
      return { kind: "pattern", regex: new RegExp(body, flags.replace(/[gy]/g, "")) };
    } catch {
      return { kind: "invalid", source: trimmedQuery };
    }
  }

  if (trimmedQuery.includes("*")) {
    const escapedGlob = trimmedQuery.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
    return { kind: "pattern", regex: new RegExp(`^${escapedGlob.replace(/\*/g, ".*")}$`, "i") };
  }

  return { kind: "substring", needle: trimmedQuery.toLowerCase() };
}

function keyActivityFields(apiKey: string, data: ModelActivityData): readonly (string | null | undefined)[] {
  const meta = data.key_metadata;
  return [apiKey, data.label, meta?.key_alias, meta?.user_id, meta?.user_email];
}

function matchesQuery(fields: readonly (string | null | undefined)[], query: KeyQuery): boolean {
  switch (query.kind) {
    case "all":
      return true;
    case "substring":
      return fields.some((field) => field?.toLowerCase().includes(query.needle) ?? false);
    case "pattern":
      return fields.some((field) => typeof field === "string" && query.regex.test(field));
    case "invalid":
      return false;
    default: {
      const unreachableQuery: never = query;
      return unreachableQuery;
    }
  }
}

export function keyActivityMatches(apiKey: string, data: ModelActivityData, query: string): boolean {
  return matchesQuery(keyActivityFields(apiKey, data), parseKeyQuery(query));
}

export function filterKeyActivity(
  keyMetrics: Record<string, ModelActivityData>,
  query: string,
): Record<string, ModelActivityData> {
  const parsedQuery = parseKeyQuery(query);
  if (parsedQuery.kind === "all") return keyMetrics;

  return Object.fromEntries(
    Object.entries(keyMetrics).filter(([apiKey, data]) => matchesQuery(keyActivityFields(apiKey, data), parsedQuery)),
  );
}
