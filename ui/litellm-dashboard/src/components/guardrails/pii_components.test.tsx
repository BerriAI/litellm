import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { CategoryFilter, QuickActions, PiiEntityList } from "./pii_components";
import type { PiiEntityCategory } from "./types";

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
