import { createParser, createSerializer } from "nuqs";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";

export const safeCustomerValue = (value: string): string | null => {
  const validLength = value.length > 0 && value.length <= 1024;
  const printable = Array.from(value).every((char) => char.charCodeAt(0) >= 32 && char.charCodeAt(0) !== 127);
  return validLength && value === value.trim() && printable ? value : null;
};

const customerId = createParser({ parse: safeCustomerValue, serialize: (value: string) => value });

export const CUSTOMER_CONTEXT_PARSERS = { customerOrg: customerId, customerMember: customerId };

const serializeCustomer = createSerializer(CUSTOMER_CONTEXT_PARSERS);

export const customerPlaygroundHref = (customerOrg: string, customerMember: string | null = null): string =>
  serializeCustomer(uiHref("playground"), { customerOrg, customerMember });

export type CustomerRequestFailure = Readonly<{ status: number | null; message: string }>;

const record = (value: unknown): Readonly<Record<string, unknown>> | null =>
  value !== null && typeof value === "object" ? (value as Readonly<Record<string, unknown>>) : null;

export const customerRequestFailure = (error: unknown): CustomerRequestFailure | null => {
  if (error === null || error === undefined) return null;
  const status =
    typeof error === "object" && "status" in error && typeof error.status === "number" ? error.status : null;
  const envelope = record(error);
  const body = record(envelope?.body) ?? record(envelope?.error);
  const detail = record(body?.error) ?? body;
  const message = typeof detail?.message === "string" ? detail.message : extractProxyErrorMessage(error);
  const fallbackCode = typeof detail?.code === "string" ? detail.code : null;
  const code = typeof detail?.type === "string" ? detail.type : fallbackCode;
  return { status, message: code && !message.includes(code) ? `${code}: ${message}` : message };
};

export const remainingCustomerBudget = (spend: number, maxBudget: number | null | undefined): number | null =>
  maxBudget == null ? null : Math.max(0, maxBudget - spend);
