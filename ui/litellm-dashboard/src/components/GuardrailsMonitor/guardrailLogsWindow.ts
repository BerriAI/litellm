import moment from "moment";

export interface UtcInstantRange {
  start: string;
  end: string;
}

const UTC_INSTANT = "YYYY-MM-DDTHH:mm:ss.SSS[Z]";

/**
 * `/guardrails/usage/logs` pads a bare `YYYY-MM-DD` to a whole UTC day, so the picker's
 * local date lands the viewer's offset away from the range they chose. Milliseconds are
 * kept because the inclusive end bound otherwise drops the final 999ms of that day.
 */
export const toUtcInstantRange = (startDate: string, endDate: string): UtcInstantRange => ({
  start: moment(startDate).startOf("day").utc().format(UTC_INSTANT),
  end: moment(endDate).endOf("day").utc().format(UTC_INSTANT),
});
