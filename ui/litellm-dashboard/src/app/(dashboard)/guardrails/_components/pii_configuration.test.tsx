import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import PiiConfiguration from "./pii_configuration";

describe("PiiConfiguration", () => {
  it("should render", () => {
    render(
      <PiiConfiguration
        entities={[]}
        actions={[]}
        selectedEntities={[]}
        selectedActions={{}}
        onEntitySelect={() => {}}
        onActionSelect={() => {}}
        entityCategories={[]}
      />,
    );
    expect(screen.getByText("Configure PII Protection")).toBeInTheDocument();
  });
});
