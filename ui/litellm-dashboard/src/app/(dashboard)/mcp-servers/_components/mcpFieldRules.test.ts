import { describe, expect, it, vi } from "vitest";
import type { MountedFieldControlProps } from "@/components/common_components/MountedFormField";
import { tagsControl } from "./mcpFieldRules";

const controlWith = (value: unknown, onChange = vi.fn()): MountedFieldControlProps =>
  ({ id: "field", name: "field", value, onChange, onBlur: vi.fn() }) as unknown as MountedFieldControlProps;

// These fields were antd Selects with tokenSeparators={[","]}: a comma commits a tag, and nothing
// else does. The shadcn tag input commits the whole typed string as one custom value instead, so
// every case below pins the antd rule the migration has to preserve.
describe("tagsControl", () => {
  it("splits the comma-separated headers an admin commits as one entry", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["Authorization,X-Custom-Header"]);

    expect(onChange).toHaveBeenCalledWith(["Authorization", "X-Custom-Header"]);
  });

  it("trims the space an admin types after each comma", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["read, write, admin"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write", "admin"]);
  });

  it("keeps already-committed tags intact while splitting only the entry that needs it", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["group-a"], onChange)).onValueChange(["group-a", "group-b,group-c"]);

    expect(onChange).toHaveBeenCalledWith(["group-a", "group-b", "group-c"]);
  });

  it("drops the duplicate when a typed entry repeats a tag that is already selected", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["read"], onChange)).onValueChange(["read", "read,write"]);

    expect(onChange).toHaveBeenCalledWith(["read", "write"]);
  });

  it("preserves the interior spaces of a stdio argument, which a comma alone may split", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["npx"], onChange)).onValueChange(["npx", "--header=X-Trace: on"]);

    expect(onChange).toHaveBeenCalledWith(["npx", "--header=X-Trace: on"]);
  });

  it("keeps a scope URI whole, since it carries no comma", () => {
    const onChange = vi.fn();
    tagsControl(controlWith([], onChange)).onValueChange(["api://app-id/.default"]);

    expect(onChange).toHaveBeenCalledWith(["api://app-id/.default"]);
  });

  it("splits a stored delimited string rather than rendering it as one tag", () => {
    expect(tagsControl(controlWith("Authorization,X-Custom-Header")).value).toStrictEqual([
      "Authorization",
      "X-Custom-Header",
    ]);
  });

  it("renders a stored list unchanged", () => {
    expect(tagsControl(controlWith(["run", "--flag a"])).value).toStrictEqual(["run", "--flag a"]);
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
