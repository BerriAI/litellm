import moment from "moment";

export interface UtcInstantRange {
  start: string;
  end: string;
}

/**
 * `/guardrails/usage/logs` filters `start_time`, a real timestamp column, but pads a
 * bare `YYYY-MM-DD` out to `T00:00:00+00:00` / `T23:59:59+00:00`. Sending the picker's
 * local calendar date therefore shifts the window by the viewer's UTC offset. Resolving
 * the local day here and sending instants keeps the bound exact; the endpoint already
 * parses a value carrying a `T`.
 */
export const toUtcInstantRange = (startDate: string, endDate: string): UtcInstantRange => ({
  start: moment(startDate).startOf("day").utc().format(),
  end: moment(endDate).endOf("day").utc().format(),
});
