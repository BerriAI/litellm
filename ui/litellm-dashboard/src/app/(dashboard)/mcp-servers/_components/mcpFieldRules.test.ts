import { describe, expect, it, vi } from "vitest";
import type { MountedFieldControlProps } from "@/components/common_components/MountedFormField";
import { tagsControl } from "./mcpFieldRules";

const controlWith = (value: unknown, onChange = vi.fn()): MountedFieldControlProps =>
  ({ id: "scopes", name: "scopes", value, onChange, onBlur: vi.fn() }) as unknown as MountedFieldControlProps;

describe("tagsControl", () => {
  it("splits a stored delimited string into one tag per scope", () => {
    expect(tagsControl(controlWith("read write admin")).value).toStrictEqual(["read", "write", "admin"]);
  });

  it("splits a comma-delimited entry the tag input committed as a single custom value", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["read,write"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write"]);
  });

  it("keeps already-split tags intact while splitting only the entry that needs it", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["read"], onChange)).onValueChange(["read", "write, admin"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write", "admin"]);
  });

  it("drops the duplicate when a typed entry repeats a tag that is already selected", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["read"], onChange)).onValueChange(["read", "read,write"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write"]);
  });

  it("clears to an empty list rather than a blank tag when the field is emptied", () => {
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
