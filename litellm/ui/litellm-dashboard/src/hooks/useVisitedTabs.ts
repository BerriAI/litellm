import { useCallback, useState } from "react";

export function useVisitedTabs(initialTab: string) {
  const [visited, setVisited] = useState<ReadonlySet<string>>(() => new Set([initialTab]));

  const onTabChange = useCallback((value: unknown) => {
    setVisited((previous) => new Set(previous).add(String(value)));
  }, []);

  const hasVisited = useCallback((value: string) => visited.has(value), [visited]);

  return { onTabChange, hasVisited };
}
