"use client";

import { parseAsString, useQueryState } from "nuqs";
import { useCallback } from "react";

export const BUILDER_INSIGHTS_PARSERS = { builder: parseAsString.withOptions({ history: "push" }) };

export function useBuilderInsightsRoute() {
  const [builder, setBuilder] = useQueryState("builder", BUILDER_INSIGHTS_PARSERS.builder);
  return {
    builderId: builder,
    selectBuilder: useCallback((id: string) => void setBuilder(id), [setBuilder]),
    closeBuilder: useCallback(() => void setBuilder(null), [setBuilder]),
  };
}
