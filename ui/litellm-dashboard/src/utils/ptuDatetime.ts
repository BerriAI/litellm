import dayjs, { Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

const WALL_CLOCK_FORMAT = "YYYY-MM-DDTHH:mm:ss";

/**
 * Read the wall clock the picker is showing and stamp it as UTC.
 *
 * The picker value stays in UTC mode end to end (see `utcIsoToPickerValue`), so `.format()`
 * returns the digits the operator sees and no zone conversion happens on the way out.
 */
export const ptuPickerToUtcIso = (value: Dayjs | null | undefined): string | null => {
  if (!value || typeof value.format !== "function") {
    return null;
  }
  // A value that came from storage is already in UTC mode and holds the exact stored
  // instant. Every save re-sends both window fields, so routing it back through a
  // second-granularity wall clock would silently drop any sub-second component of a
  // window that was set out of band, turning an unrelated edit into a quiet rewrite.
  if (typeof value.isUTC === "function" && value.isUTC()) {
    return value.toISOString();
  }
  // A freshly picked value is in the browser's zone; its wall clock is what the operator
  // chose against a UTC-labelled field, so it is reinterpreted rather than converted.
  return dayjs.utc(value.format(WALL_CLOCK_FORMAT)).toISOString();
};

/**
 * Hand the picker a UTC-mode Dayjs so it displays the stored wall clock verbatim.
 *
 * Re-parsing the wall clock in the browser's zone looks equivalent but is not: a clock reading
 * that does not exist locally, the hour a DST spring-forward skips, gets advanced by the engine.
 * `dayjs("2027-03-14T02:30:00")` is 03:30 in America/Los_Angeles, and because the save path
 * re-stamps whatever the picker holds as UTC, that shift would be written back to the stored
 * window rather than cancelled out.
 */
export const utcIsoToPickerValue = (iso: string | null | undefined): Dayjs | null => {
  if (!iso) {
    return null;
  }
  const parsed = dayjs.utc(iso);
  return parsed.isValid() ? parsed : null;
};

const DISPLAY_FORMAT = "YYYY-MM-DD HH:mm:ss";

/**
 * Render a stored PTU timestamp for the read view. The backend serialises as `+00:00` while a
 * just-saved form holds the `.000Z` the picker produced, so the same instant would otherwise be
 * shown two different ways depending on whether the page has been reloaded since the edit. An
 * unparseable value is passed through rather than hidden.
 */
export const formatPtuUtcDisplay = (iso: string | null | undefined): string | null => {
  if (!iso) {
    return null;
  }
  const parsed = dayjs.utc(iso);
  return parsed.isValid() ? `${parsed.format(DISPLAY_FORMAT)} UTC` : String(iso);
};
