import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { RequestTagMapping, RequestTagMappingsEditor } from "./RequestTagMappings";

const exact: RequestTagMapping = { header: "baggage", key: "entrypoint", value: "a", tag: "support" };
const variable: RequestTagMapping = { header: "baggage", key: "entrypoint", tag_prefix: "usecase:" };

function Editor({ initial, changed }: { initial: RequestTagMapping[]; changed: (rules: RequestTagMapping[]) => void }) {
  const [rules, setRules] = useState(initial);
  return (
    <RequestTagMappingsEditor
      value={rules}
      onChange={(next) => {
        setRules(next);
        changed(next);
      }}
    />
  );
}

describe("RequestTagMappingsEditor", () => {
  it("shows YAML-owned rules without allowing edits", () => {
    render(<RequestTagMappingsEditor value={[exact]} onChange={vi.fn()} disabled />);
    expect(screen.getByLabelText("Mapping 1 match value")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add mapping" })).toBeDisabled();
    expect(screen.getByText(/Managed in config.yaml/)).toBeInTheDocument();
  });
  it("shows one-by-one mappings alongside variable prefix mappings", () => {
    render(<Editor initial={[exact, variable]} changed={vi.fn()} />);
    expect(screen.getByLabelText("Mapping 1 match value")).toHaveValue("a");
    expect(screen.getByLabelText("Mapping 1 tag")).toHaveValue("support");
    expect(screen.getByLabelText("Mapping 2 tag prefix")).toHaveValue("usecase:");
    expect(screen.getByText("entrypoint=a → support")).toBeInTheDocument();
    expect(screen.getByText("entrypoint=x → usecase:x")).toBeInTheDocument();
  });

  it("adds and edits a new exact value mapping", async () => {
    const changed = vi.fn();
    render(<Editor initial={[]} changed={changed} />);
    await userEvent.click(screen.getByRole("button", { name: "Add mapping" }));
    fireEvent.change(screen.getByLabelText("Mapping 1 match value"), { target: { value: "b" } });
    fireEvent.change(screen.getByLabelText("Mapping 1 tag"), { target: { value: "sales" } });
    expect(screen.getByText("entrypoint=b → sales")).toBeInTheDocument();
    expect(changed).toHaveBeenLastCalledWith([{ header: "baggage", key: "entrypoint", value: "b", tag: "sales" }]);
  });

  it("changes an exact mapping to any-value mode and removes its exact match fields", async () => {
    const changed = vi.fn();
    render(<Editor initial={[exact]} changed={changed} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Mapping 1 mode" }));
    await userEvent.click(await screen.findByRole("option", { name: "Any value" }));
    expect(screen.queryByLabelText("Mapping 1 match value")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Mapping 1 tag prefix")).toHaveValue("usecase:");
    expect(changed).toHaveBeenLastCalledWith([
      expect.objectContaining({ value: undefined, tag: undefined, tag_prefix: "usecase:" }),
    ]);
  });

  it("deletes one rule while keeping the other mode intact", async () => {
    const changed = vi.fn();
    render(<Editor initial={[exact, variable]} changed={changed} />);
    await userEvent.click(screen.getByRole("button", { name: "Delete mapping 1" }));
    expect(screen.getByLabelText("Mapping 1 tag prefix")).toHaveValue("usecase:");
    expect(changed).toHaveBeenLastCalledWith([variable]);
  });
});
