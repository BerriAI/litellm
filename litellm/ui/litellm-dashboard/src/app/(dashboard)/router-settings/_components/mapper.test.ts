import { describe, expect, it } from "vitest";
import type { generalSettingsItem } from "./general_settings";
import { resetGeneralSetting, toGeneralSettingUpdate } from "./mapper";

const row = (field_name: string, field_type: string, field_value: unknown): generalSettingsItem => ({
  field_name,
  field_type,
  field_value,
  field_description: "",
  stored_in_db: true,
});

describe("General settings row adapter", () => {
  it.each([0, false, "", ["first", "second"], { header: "value" }].map((value) => ({ value })))(
    "submits the raw value $value",
    ({ value }) => {
      expect(toGeneralSettingUpdate([row("setting", "List", value)], "setting")).toEqual({ kind: "update", value });
    },
  );

  it("resets a nullish Select or List without submitting it and skips other nullish or missing values", () => {
    const rows = [row("select", "Select", null), row("list", "List", undefined), row("number", "Integer", null)];

    expect(toGeneralSettingUpdate(rows, "select")).toEqual({ kind: "reset" });
    expect(toGeneralSettingUpdate(rows, "list")).toEqual({ kind: "reset" });
    expect(toGeneralSettingUpdate(rows, "number")).toEqual({ kind: "skip" });
    expect(toGeneralSettingUpdate(rows, "missing")).toEqual({ kind: "skip" });
  });

  it("keeps row order and restores the raw default while marking only the reset row as not set", () => {
    const first = row("first", "Boolean", "false");
    const second = { ...row("second", "Integer", 5), field_default_value: 0 };
    const third = row("third", "List", ["third"]);
    const rows = [first, second, third];

    expect(resetGeneralSetting(rows, "second").items).toEqual([
      first,
      { ...second, field_value: 0, stored_in_db: null },
      third,
    ]);
    expect(resetGeneralSetting(rows, "third").items).toEqual([
      first,
      second,
      { ...third, field_value: null, stored_in_db: null },
    ]);
  });
});
