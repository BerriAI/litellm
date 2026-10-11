import { $api } from "@/lib/http/api";
import { latestLaunches } from "./homeContent";

export const useWhatsNew = () =>
  $api.useQuery(
    "get",
    "/public/whats_new",
    {},
    { staleTime: 60 * 60 * 1000, select: (data) => latestLaunches(data.launches) },
  );
