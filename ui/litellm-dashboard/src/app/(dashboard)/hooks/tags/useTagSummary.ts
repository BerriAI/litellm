import { $api } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const isoDay = (date: Date) => {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
};

/** Per-tag spend, tokens and requests for a date range, including the `User-Agent:` tags agents are read from. */
export const useTagSummary = (startTime: Date | null, endTime: Date | null, enabled = true) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    "/tag/summary",
    {
      params: {
        query: {
          start_date: startTime ? isoDay(startTime) : "",
          end_date: endTime ? isoDay(endTime) : "",
        },
      },
    },
    {
      enabled: enabled && Boolean(accessToken && startTime && endTime),
      select: (data) => data.results,
    },
  );
};
