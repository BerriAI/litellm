import { useSyncExternalStore } from "react";

import { narrow as CATALOG_NARROW_QUERY } from "./catalogBreakpoints.json";

const subscribe = (onChange: () => void) => {
  const query = window.matchMedia(CATALOG_NARROW_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
};

export const useIsNarrow = (): boolean =>
  useSyncExternalStore(
    subscribe,
    () => window.matchMedia(CATALOG_NARROW_QUERY).matches,
    () => false,
  );
