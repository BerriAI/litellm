import { expectTypeOf, test } from "vitest";

import type { components, paths } from "@/lib/http/schema";

type SQLResponse = paths["/v1/traces/query"]["post"]["responses"][200]["content"]["application/json"];
type QueryHelp = paths["/v1/traces/query/help"]["get"]["responses"][200]["content"]["application/json"];

test("SQL results expose only row data and JSON values", () => {
  expectTypeOf<keyof SQLResponse>().toEqualTypeOf<"data">();
  expectTypeOf<SQLResponse["data"][number][string]>().toEqualTypeOf<components["schemas"]["JsonValue"]>();
});

test("query help exposes schema discovery, examples and discovery failures", () => {
  expectTypeOf<QueryHelp["tables"][number]["columns"][number]["type"]>().toEqualTypeOf<string>();
  expectTypeOf<QueryHelp["metadata"]["error"]>().toEqualTypeOf<string | null | undefined>();
  expectTypeOf<QueryHelp["examples"][number]["sql"]>().toEqualTypeOf<string>();
  expectTypeOf<QueryHelp["guide"]>().toEqualTypeOf<string>();
});
