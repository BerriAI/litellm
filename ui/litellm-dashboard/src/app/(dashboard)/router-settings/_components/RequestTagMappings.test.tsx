import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { RequestTagMapping, RequestTagMappingsEditor, requestTagMappingsAreValid } from "./RequestTagMappings";

const exact: RequestTagMapping = { header: "baggage", key: "entrypoint", value: "a", tag: "support" };
const variable: RequestTagMapping = { header: "baggage", key: "entrypoint", tag_prefix: "usecase:" };

function Editor({
  initial,
  changed,
  saved = vi.fn(),
}: {
  initial: RequestTagMapping[];
  changed: (rules: RequestTagMapping[]) => void;
  saved?: (rules: RequestTagMapping[]) => void;
}) {
  const [rules, setRules] = useState(initial);
  return (
    <>
      <RequestTagMappingsEditor
        value={rules}
        onChange={(next) => {
          setRules(next);
          changed(next);
        }}
      />
      <button type="button" disabled={!requestTagMappingsAreValid(rules)} onClick={() => saved(rules)}>
        Update
      </button>
    </>
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
    expect(screen.getByRole("combobox", { name: "Mapping 1 mode" })).toHaveTextContent("Exact value");
    expect(screen.getByRole("combobox", { name: "Mapping 2 mode" })).toHaveTextContent("Any value");
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
    fireEvent.change(screen.getByLabelText("Mapping 1 baggage key"), { target: { value: "entrypoint" } });
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
    expect(screen.getByLabelText("Mapping 1 tag prefix")).toHaveValue("");
    expect(screen.getByLabelText("Mapping 1 tag prefix")).toHaveAttribute("placeholder", "usecase:");
    expect(changed).toHaveBeenLastCalledWith([
      expect.objectContaining({ value: undefined, tag: undefined, tag_prefix: "" }),
    ]);
  });

  it("blocks saving a new mapping until its match value and tag are filled in", async () => {
    const saved = vi.fn();
    render(<Editor initial={[]} changed={vi.fn()} saved={saved} />);
    await userEvent.click(screen.getByRole("button", { name: "Add mapping" }));
    expect(screen.getByLabelText("Mapping 1 baggage key")).toHaveValue("");
    expect(screen.getByLabelText("Mapping 1 baggage key")).toHaveAttribute("placeholder", "entrypoint");
    expect(screen.getByText("Enter a baggage key")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Mapping 1 baggage key"), { target: { value: "entrypoint" } });
    expect(screen.getByText("Enter a match value")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Update" }));
    fireEvent.change(screen.getByLabelText("Mapping 1 match value"), { target: { value: "b" } });
    expect(screen.getByText("Enter a tag")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Update" }));
    expect(saved).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Mapping 1 tag"), { target: { value: "sales" } });
    expect(screen.queryByText(/^Enter a/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Update" }));
    expect(saved).toHaveBeenCalledWith([{ header: "baggage", key: "entrypoint", value: "b", tag: "sales" }]);
  });

  it("blocks saving an any-value mapping until the admin types a tag prefix", async () => {
    const saved = vi.fn();
    render(<Editor initial={[exact]} changed={vi.fn()} saved={saved} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Mapping 1 mode" }));
    await userEvent.click(await screen.findByRole("option", { name: "Any value" }));
    expect(screen.getByText("Enter a tag prefix")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Mapping 1 tag prefix"), { target: { value: "team:" } });
    await userEvent.click(screen.getByRole("button", { name: "Update" }));
    expect(saved).toHaveBeenCalledWith([expect.objectContaining({ key: "entrypoint", tag_prefix: "team:" })]);
  });

  it("blocks saving a baggage key that is empty or not a header token", async () => {
    render(<Editor initial={[exact]} changed={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Mapping 1 baggage key"), { target: { value: "" } });
    expect(screen.getByText("Enter a baggage key")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Mapping 1 baggage key"), { target: { value: "entry point" } });
    expect(screen.getByText(/Use only letters, digits/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Mapping 1 baggage key"), { target: { value: "entry.point" } });
    expect(screen.getByRole("button", { name: "Update" })).toBeEnabled();
  });

  it("blocks saving a second mapping for the same key and value until it differs", () => {
    render(<Editor initial={[exact, { ...exact, tag: "sales" }, variable]} changed={vi.fn()} />);
    expect(screen.getByText("Another mapping already matches this key and value")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Mapping 2 match value"), { target: { value: "b" } });
    expect(screen.queryByText("Another mapping already matches this key and value")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeEnabled();
  });

  it("blocks saving a second any-value mapping for the same key", () => {
    render(<Editor initial={[variable, { ...variable, tag_prefix: "team:" }]} changed={vi.fn()} />);
    expect(screen.getByText("Another mapping already matches this key and value")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
  });

  it("deletes one rule while keeping the other mode intact", async () => {
    const changed = vi.fn();
    render(<Editor initial={[exact, variable]} changed={changed} />);
    await userEvent.click(screen.getByRole("button", { name: "Delete mapping 1" }));
    expect(screen.getByLabelText("Mapping 1 tag prefix")).toHaveValue("usecase:");
    expect(changed).toHaveBeenLastCalledWith([variable]);
  });
});
