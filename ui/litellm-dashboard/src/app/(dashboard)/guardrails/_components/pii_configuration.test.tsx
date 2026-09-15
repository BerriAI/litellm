import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
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

  it("renders selected custom entities", () => {
    render(
      <PiiConfiguration
        entities={["PERSON"]}
        actions={["MASK"]}
        selectedEntities={["NO_FODSELSNUMMER"]}
        selectedActions={{ NO_FODSELSNUMMER: "MASK" }}
        onEntitySelect={() => {}}
        onActionSelect={() => {}}
      />,
    );

    expect(screen.getByText("NO FODSELSNUMMER")).toBeInTheDocument();
  });

  it("adds a normalized custom entity with the first action", () => {
    const onEntitySelect = vi.fn();
    const onActionSelect = vi.fn();
    render(
      <PiiConfiguration
        entities={["PERSON"]}
        actions={["MASK", "BLOCK"]}
        selectedEntities={[]}
        selectedActions={{}}
        onEntitySelect={onEntitySelect}
        onActionSelect={onActionSelect}
      />,
    );

    fireEvent.change(screen.getByRole("textbox", { name: "Custom entity name" }), {
      target: { value: "se personnummer" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add entity" }));

    expect(onEntitySelect).toHaveBeenCalledWith("SE_PERSONNUMMER");
    expect(onActionSelect).toHaveBeenCalledWith("SE_PERSONNUMMER", "MASK");
  });

  it("rejects invalid custom entity names", () => {
    const onEntitySelect = vi.fn();
    const onActionSelect = vi.fn();
    render(
      <PiiConfiguration
        entities={["PERSON"]}
        actions={["MASK", "BLOCK"]}
        selectedEntities={[]}
        selectedActions={{}}
        onEntitySelect={onEntitySelect}
        onActionSelect={onActionSelect}
      />,
    );

    fireEvent.change(screen.getByRole("textbox", { name: "Custom entity name" }), {
      target: { value: "bad!name" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add entity" }));

    expect(screen.getByText("Use letters, numbers and underscores only")).toBeInTheDocument();
    expect(onEntitySelect).not.toHaveBeenCalled();
    expect(onActionSelect).not.toHaveBeenCalled();
  });
});
