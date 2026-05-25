import moment from "moment";

// IST is UTC+05:30. We use moment's utcOffset (no moment-timezone dependency) so
// these helpers behave the same regardless of the browser's local timezone.
export const IST_OFFSET = "+05:30";

// Current time expressed as an IST wall-clock datetime-local string (for inputs).
export const nowISTLocal = (subtract?: {
  value: number;
  unit: moment.unitOfTime.DurationConstructor;
}): string => {
  const m = moment().utcOffset(IST_OFFSET);
  if (subtract) m.subtract(subtract.value, subtract.unit);
  return m.format("YYYY-MM-DDTHH:mm");
};

// Interpret an IST datetime-local string as IST wall time, return a UTC string for the API.
export const istLocalToUtc = (istLocal: string): string =>
  moment(istLocal, "YYYY-MM-DDTHH:mm").utcOffset(IST_OFFSET, true).utc().format("YYYY-MM-DD HH:mm:ss");

// Render a UTC ISO timestamp from the backend as an IST wall-clock string.
export const utcToISTDisplay = (utcIso: string): string =>
  moment.utc(utcIso).utcOffset(IST_OFFSET).format("YYYY-MM-DD HH:mm:ss.SSS [IST]");
