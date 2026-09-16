import { useState } from "react";

export const useVisitedTabs = <T extends string>(activeTab: T): readonly T[] => {
  const [visitedTabs, setVisitedTabs] = useState<readonly T[]>([activeTab]);
  if (!visitedTabs.includes(activeTab)) {
    setVisitedTabs([...visitedTabs, activeTab]);
  }
  return visitedTabs;
};
