import { type DisableableCapability } from "./schema";

export interface AllowedClient {
  readonly alias: string;
  readonly value: string;
}

const isAllowedClient = (entry: unknown): entry is AllowedClient => {
  if (typeof entry !== "object" || entry === null) return false;
  const { alias, value } = entry as Partial<Record<keyof AllowedClient, unknown>>;
  return typeof alias === "string" && typeof value === "string" && !isIncomplete({ alias, value });
};

export type StoredAllowlist =
  | { readonly kind: "absent" }
  | { readonly kind: "clients"; readonly clients: AllowedClient[] }
  | { readonly kind: "malformed" };

export const ABSENT: StoredAllowlist = { kind: "absent" };

export const toAllowedClients = (rows: AllowedClient[]): AllowedClient[] =>
  rows.map(({ alias, value }) => ({ alias, value }));

export const parseStoredClients = (fieldValue: unknown): StoredAllowlist => {
  if (fieldValue === null || fieldValue === undefined) return ABSENT;
  if (Array.isArray(fieldValue) && fieldValue.every(isAllowedClient)) {
    return { kind: "clients", clients: toAllowedClients(fieldValue) };
  }
  return { kind: "malformed" };
};

export const trimClient = ({ alias, value }: AllowedClient): AllowedClient => ({
  alias: alias.trim(),
  value: value.trim(),
});

export const isIncomplete = ({ alias, value }: AllowedClient) => alias === "" || value === "";

export const unchangedSinceLoad = (value: string[], stored: string[] | null) =>
  stored === null
    ? value.length === 0
    : value.length > 0 && value.length === stored.length && value.every((entry, i) => entry === stored[i]);

export const clientsUnchangedSinceLoad = (value: AllowedClient[], stored: StoredAllowlist) => {
  switch (stored.kind) {
    case "absent":
      return value.length === 0;
    case "clients":
      return (
        value.length > 0 &&
        value.length === stored.clients.length &&
        value.every((client, i) => client.alias === stored.clients[i].alias && client.value === stored.clients[i].value)
      );
    case "malformed":
      return false;
  }
};

export const headerUnchangedSinceLoad = (value: string, stored: string | null) =>
  stored === null ? value === "" : value !== "" && value === stored;

export const parseDisabledCapabilities = (value: unknown): DisableableCapability[] =>
  Array.isArray(value)
    ? value.filter((name): name is DisableableCapability => name === "prompts" || name === "resources")
    : [];
