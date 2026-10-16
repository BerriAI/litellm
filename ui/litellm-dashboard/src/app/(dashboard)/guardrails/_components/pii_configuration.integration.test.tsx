import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import PiiConfiguration from "./pii_configuration";

const StatefulPiiConfiguration = () => {
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{ [key: string]: string }>({});

  return (
    <PiiConfiguration
      entities={["PERSON", "EMAIL"]}
      actions={["MASK", "BLOCK"]}
      selectedEntities={selectedEntities}
      selectedActions={selectedActions}
      onEntitySelect={(entity) => {
        setSelectedEntities((previous) =>
          previous.includes(entity) ? previous.filter((selected) => selected !== entity) : [...previous, entity],
        );
      }}
      onActionSelect={(entity, action) => {
        setSelectedActions((previous) => ({ ...previous, [entity]: action }));
      }}
      entityCategories={[
        { category: "Personal", entities: ["PERSON"] },
        { category: "Contact", entities: ["EMAIL"] },
      ]}
    />
  );
};

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

  it("adds a normalized custom entity with MASK", () => {
    const onEntitySelect = vi.fn();
    const onActionSelect = vi.fn();
    render(
      <PiiConfiguration
        entities={["PERSON"]}
        actions={["BLOCK", "MASK"]}
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

  it("shows a newly added custom entity while a category filter is active", async () => {
    const user = userEvent.setup();
    render(<StatefulPiiConfiguration />);

    const categoryInput = screen.getByPlaceholderText("Select categories to filter by");
    await user.click(categoryInput);
    if (categoryInput.getAttribute("aria-expanded") !== "true") {
      categoryInput.focus();
      await user.keyboard("{Enter}");
    }
    await user.click(await screen.findByRole("option", { name: "Personal" }));
    await user.keyboard("{Escape}");

    expect(screen.getByText("PERSON")).toBeVisible();
    expect(screen.queryByText("EMAIL")).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Custom entity name" }), {
      target: { value: "se personnummer" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add entity" }));

    expect(await screen.findByText("SE PERSONNUMMER")).toBeVisible();
    expect(screen.getByText("PERSON")).toBeVisible();
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
