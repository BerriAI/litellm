import { describe, expectTypeOf, test } from "vitest";

import type { components } from "@/lib/http/schema";
import { CLEARED_SSO_SETTINGS } from "./utils";

describe("SSO settings utils types", () => {
  test("clearing SSO settings sends null for exactly the fields the backend declares", () => {
    expectTypeOf(CLEARED_SSO_SETTINGS).toEqualTypeOf<{
      readonly [K in keyof components["schemas"]["SSOConfig"]]-?: null;
    }>();
  });
});
