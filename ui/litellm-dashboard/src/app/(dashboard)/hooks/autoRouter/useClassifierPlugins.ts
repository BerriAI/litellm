import { $api } from "@/lib/http/api";

export const useClassifierPlugins = () =>
  $api.useQuery(
    "get",
    "/auto_router/classifier_plugins",
    {},
    {
      // Registration is config-file-only, so the list changes only on a proxy reload.
      staleTime: 5 * 60 * 1000,
      select: (data) => data.classifier_plugins,
    },
  );
