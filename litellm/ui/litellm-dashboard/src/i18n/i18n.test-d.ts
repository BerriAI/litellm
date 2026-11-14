import type { TFunction } from "i18next";
import { expectTypeOf, test } from "vitest";

test("translation keys come from the English message file", () => {
  const t = {} as TFunction;

  expectTypeOf(t("workflows.drawer.eventCount", { count: 2 })).toBeString();
  // @ts-expect-error
  t("workflows.header.titel");
});
