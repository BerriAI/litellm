import type { KeyValuePair } from "@/components/key_value_input";

export type CustomHeaders = Readonly<Record<string, string>>;

export const CUSTOMER_ORG_HEADER = "x-litellm-customer-org-alias";
export const CUSTOMER_MEMBER_HEADER = "x-litellm-customer-member-id";

export const customerHeaderPairs = (
  pairs: readonly KeyValuePair[],
  alias: string | null,
  memberId: string | null,
): readonly KeyValuePair[] => [
  ...pairs.filter(([name]) => ![CUSTOMER_ORG_HEADER, CUSTOMER_MEMBER_HEADER].includes(name.trim().toLowerCase())),
  ...(alias ? [[CUSTOMER_ORG_HEADER, alias] as KeyValuePair] : []),
  ...(alias && memberId ? [[CUSTOMER_MEMBER_HEADER, memberId] as KeyValuePair] : []),
];

export const customerFromHeaderPairs = (pairs: readonly KeyValuePair[]) => ({
  alias: pairs.findLast(([name]) => name.trim().toLowerCase() === CUSTOMER_ORG_HEADER)?.[1] || null,
  memberId: pairs.findLast(([name]) => name.trim().toLowerCase() === CUSTOMER_MEMBER_HEADER)?.[1] || null,
});

export const customHeadersFromPairs = (pairs: readonly KeyValuePair[]): CustomHeaders =>
  Object.fromEntries(pairs.map(([name, value]) => [name.trim(), value]).filter(([name]) => name !== ""));

const isHeaderPair = (entry: unknown): entry is KeyValuePair =>
  Array.isArray(entry) && entry.length === 2 && entry.every((part) => typeof part === "string");

export const parseStoredHeaderPairs = (raw: string | null): readonly KeyValuePair[] => {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isHeaderPair) : [];
  } catch {
    return [];
  }
};

export const buildPlaygroundHeaders = (
  tags?: readonly string[],
  customHeaders?: CustomHeaders,
): Record<string, string> => ({
  ...(tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : {}),
  ...customHeaders,
});

export const withRequiredHeaders = (
  headers: Readonly<Record<string, string>>,
  required: Readonly<Record<string, string>>,
): Record<string, string> => {
  const reserved = new Set(Object.keys(required).map((name) => name.toLowerCase()));
  return {
    ...Object.fromEntries(Object.entries(headers).filter(([name]) => !reserved.has(name.toLowerCase()))),
    ...required,
  };
};
