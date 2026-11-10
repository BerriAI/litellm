import { describe, expect, it, vi } from "vitest";
import type { MountedFieldControlProps } from "@/components/common_components/MountedFormField";
import { tagsControl } from "./mcpFieldRules";

const controlWith = (value: unknown, onChange = vi.fn()): MountedFieldControlProps =>
  ({ id: "field", name: "field", value, onChange, onBlur: vi.fn() }) as unknown as MountedFieldControlProps;

// These fields were antd Selects with tokenSeparators={[","]}, so a comma commits a tag as an admin
// types. MultiSelect owns that rule now, which leaves this adapter one job: hand the stored value to
// the input and the edited value back, without rewriting either. Stdio args are process argv, so a
// comma inside one and a deliberately repeated flag both have to survive a round trip.
describe("tagsControl", () => {
  it("keeps a stored argument that contains a comma as one argument", () => {
    expect(tagsControl(controlWith(["--filter=a,b", "--verbose"])).value).toStrictEqual(["--filter=a,b", "--verbose"]);
  });

  it("keeps a repeated stdio flag rather than collapsing it to one", () => {
    expect(tagsControl(controlWith(["-v", "-v"])).value).toStrictEqual(["-v", "-v"]);
  });

  it("offers each repeated tag once, since the dropdown keys its entries by value", () => {
    expect(tagsControl(controlWith(["-v", "-v"])).options).toStrictEqual([{ label: "-v", value: "-v" }]);
  });

  it("stores the edited tags exactly as the input committed them", () => {
    const onChange = vi.fn();
    tagsControl(controlWith(["npx"], onChange)).onValueChange(["npx", "--header=X-Trace: on"]);

    expect(onChange).toHaveBeenCalledWith(["npx", "--header=X-Trace: on"]);
  });

  it("renders a stored list unchanged", () => {
    expect(tagsControl(controlWith(["run", "--flag a"])).value).toStrictEqual(["run", "--flag a"]);
  });

  it("clears to an empty list rather than a blank tag when the field is emptied", () => {
    expect(tagsControl(controlWith("")).value).toStrictEqual([]);
    expect(tagsControl(controlWith([""])).value).toStrictEqual([]);
  });

  it("ignores a stored value that is neither a string nor a list", () => {
    expect(tagsControl(controlWith(null)).value).toStrictEqual([]);
    expect(tagsControl(controlWith(42)).value).toStrictEqual([]);
  });

  it("wraps a bare stored string into the single tag the antd field would have shown", () => {
    expect(tagsControl(controlWith("Authorization")).value).toStrictEqual(["Authorization"]);
  });
});
