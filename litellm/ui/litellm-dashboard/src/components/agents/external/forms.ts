import { createParser } from "nuqs";

export const pageParser = createParser({
  parse: (value) => (/^[1-9]\d*$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : null),
  serialize: String,
}).withDefault(1);

type MappingResult =
  | { ok: true; mappings: { external_actor_id: string; user_id: string }[] }
  | { ok: false; error: string };

export function memberMappings(csv: string): MappingResult {
  const lines = csv
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim());
  if (!lines.length || lines.length > 1000) return { ok: false, error: "Paste between 1 and 1,000 member mappings." };
  const rows = lines.map((line) => line.split(",").map((field) => field.trim()));
  const invalid = rows.findIndex((fields) => fields.length !== 2 || !fields[0] || !fields[1]);
  if (invalid >= 0) return { ok: false, error: `Line ${invalid + 1} must contain grokbot_member_id,litellm_user_id.` };
  return { ok: true, mappings: rows.map((fields) => ({ external_actor_id: fields[0], user_id: fields[1] })) };
}

export const displayTime = (value?: string | null) => (value ? new Date(value).toLocaleString() : "No events yet");
