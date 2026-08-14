import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import PiiConfiguration from "./pii_configuration";

describe("PiiConfiguration", () => {
  it("should render", () => {
    const { getByText } = render(
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
    expect(getByText("Configure PII Protection")).toBeInTheDocument();
  });
});
