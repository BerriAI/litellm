import { describe, expect, it, vi } from "vitest";
import type { MountedFieldControlProps } from "@/components/common_components/MountedFormField";
import { scopesControl, tagsControl } from "./mcpFieldRules";

const controlWith = (value: unknown, onChange = vi.fn()): MountedFieldControlProps =>
  ({ id: "field", name: "field", value, onChange, onBlur: vi.fn() }) as unknown as MountedFieldControlProps;

describe("scopesControl", () => {
  it("splits a stored delimited string into one tag per scope", () => {
    expect(scopesControl(controlWith("read write admin")).value).toStrictEqual(["read", "write", "admin"]);
  });

  it("splits a comma-delimited entry the tag input committed as a single custom value", () => {
    const onChange = vi.fn();
    scopesControl(controlWith([], onChange)).onValueChange(["read,write"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write"]);
  });

  it("keeps already-split scopes intact while splitting only the entry that needs it", () => {
    const onChange = vi.fn();
    scopesControl(controlWith(["read"], onChange)).onValueChange(["read", "write, admin"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write", "admin"]);
  });

  it("drops the duplicate when a typed entry repeats a scope that is already selected", () => {
    const onChange = vi.fn();
    scopesControl(controlWith(["read"], onChange)).onValueChange(["read", "read,write"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write"]);
  });

  it("keeps a URI-shaped scope whole, since it carries no delimiter", () => {
    const onChange = vi.fn();
    scopesControl(controlWith([], onChange)).onValueChange(["api://app-id/.default"]);

    expect(onChange).toHaveBeenCalledWith(["api://app-id/.default"]);
  });

  it("clears to an empty list rather than a blank scope when the field is emptied", () => {
    const onChange = vi.fn();
    scopesControl(controlWith("", onChange)).onValueChange([""]);

    expect(scopesControl(controlWith("")).value).toStrictEqual([]);
    expect(onChange).toHaveBeenCalledWith([]);
  });
});

describe("tagsControl", () => {
  it("preserves a stdio argument that contains spaces, which is one argv entry", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["npx"], onChange)).onValueChange(["npx", "--header=X-Trace: on"]);

    expect(onChange).toHaveBeenCalledWith(["npx", "--header=X-Trace: on"]);
  });

  it("preserves a comma inside an entry, which is part of the value and not a separator", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["Create, update and delete issues"]);

    expect(onChange).toHaveBeenCalledWith(["Create, update and delete issues"]);
  });

  it("keeps a repeated entry, since two argv entries may legitimately be identical", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["-v", "-v"]);

    expect(onChange).toHaveBeenCalledWith(["-v", "-v"]);
  });

  it("renders a stored list unchanged", () => {
    expect(tagsControl(controlWith(["run", "--flag a"])).value).toStrictEqual(["run", "--flag a"]);
  });

  it("drops the blank entry a cleared field leaves behind", () => {
    const onChange = vi.fn();
    tagsControl(controlWith("", onChange)).onValueChange([""]);

    expect(tagsControl(controlWith("")).value).toStrictEqual([]);
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("ignores a stored value that is neither a string nor a list", () => {
    expect(tagsControl(controlWith(null)).value).toStrictEqual([]);
    expect(tagsControl(controlWith(42)).value).toStrictEqual([]);
  });
});
