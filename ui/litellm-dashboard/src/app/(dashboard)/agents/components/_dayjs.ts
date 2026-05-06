/**
 * Shared dayjs helper for the cloud-agents dashboard.
 *
 * Loads the `relativeTime` plugin once so `dayjs(...).fromNow()` is
 * typed and works across all the agents components. Without this, the
 * plugin call is untyped and TS rejects it.
 */
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

dayjs.extend(relativeTime);

export function relativeOrAbsolute(value: string | null | undefined): string {
  if (!value) return "—";
  const d = dayjs(value);
  if (!d.isValid()) return "—";
  return d.fromNow();
}

export { dayjs };
