import Papa from "papaparse";
import { z } from "zod";

export type MemberImportRow = { email: string; status: "ready" | "invalid" | "duplicate" | "existing" };

export function parseMemberEmails(text: string, existingEmails: readonly string[]): MemberImportRow[] {
  const parsed = Papa.parse<string[]>(text.trim(), { skipEmptyLines: "greedy" });
  const cells = parsed.data.flatMap((row) => row.flatMap((cell) => cell.split(/[;\s]+/))).filter(Boolean);
  const emails =
    cells[0]?.toLowerCase() === "email" || cells[0]?.toLowerCase() === "user_email" ? cells.slice(1) : cells;
  const existing = new Set(existingEmails.map((email) => email.toLowerCase()));
  return emails.map((email, index) => {
    if (!z.email({ pattern: z.regexes.html5Email }).safeParse(email).success) return { email, status: "invalid" };
    if (emails.slice(0, index).some((previous) => previous.toLowerCase() === email.toLowerCase()))
      return { email, status: "duplicate" };
    if (existing.has(email.toLowerCase())) return { email, status: "existing" };
    return { email, status: "ready" };
  });
}

export function memberImportBudget(value: string): number | undefined | "invalid" {
  if (!value.trim()) return undefined;
  const budget = Number(value);
  return Number.isFinite(budget) && budget >= 0 ? budget : "invalid";
}
