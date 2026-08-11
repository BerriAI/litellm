import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CategoryFilter, QuickActions, PiiEntityList } from "./pii_components";
import type { PiiEntityCategory } from "@/components/guardrails/types";

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  const t = (key: string, values?: Record<string, unknown>) => {
    const copy = key.split(".").reduce<unknown>((value, segment) => {
      if (typeof value !== "object" || value === null) return undefined;
      return (value as Record<string, unknown>)[segment];
    }, resources.en.gateway);
    if (typeof copy !== "string") return key;
    return Object.entries(values ?? {}).reduce(
      (text, [name, value]) => text.replaceAll(`{{${name}}}`, String(value)),
      copy,
    );
  };
  return { useTranslation: () => ({ t }) };
});

describe("CategoryFilter", () => {
  it("should render", () => {
    const emptyCategories: PiiEntityCategory[] = [];
    const { getByText } = render(
      <CategoryFilter categories={emptyCategories} selectedCategories={[]} onChange={() => {}} />,
    );
    expect(getByText("Filter by category")).toBeInTheDocument();
  });
});

describe("QuickActions", () => {
  it("should render", () => {
    const { getByText } = render(
      <QuickActions onSelectAll={() => {}} onUnselectAll={() => {}} hasSelectedEntities={false} />,
    );
    expect(getByText("Quick Actions")).toBeInTheDocument();
  });
});

describe("PiiEntityList", () => {
  it("should render", () => {
    const { getByText } = render(
      <PiiEntityList
        entities={[]}
        selectedEntities={[]}
        selectedActions={{}}
        actions={[]}
        onEntitySelect={() => {}}
        onActionSelect={() => {}}
        entityToCategoryMap={new Map()}
      />,
    );
    expect(getByText("No PII types match your filter criteria")).toBeInTheDocument();
  });
});
