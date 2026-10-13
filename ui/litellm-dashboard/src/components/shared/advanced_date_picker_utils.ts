import moment from "moment";

/** Calendar-day presets include today as the final day. */
export function getInclusiveDayRange(days: 7 | 30, now: Date = new Date()): { from: Date; to: Date } {
  return {
    from: moment(now)
      .subtract(days - 1, "days")
      .startOf("day")
      .toDate(),
    to: moment(now).endOf("day").toDate(),
  };
}
