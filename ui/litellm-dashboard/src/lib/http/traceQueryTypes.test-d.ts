import { expectTypeOf, test } from "vitest";

import type { components, paths } from "@/lib/http/schema";

type SQLResponse = paths["/v1/traces/query"]["post"]["responses"][200]["content"]["application/json"];
type QueryHelp = paths["/v1/traces/query/help"]["get"]["responses"][200]["content"]["application/json"];

test("SQL results expose column metadata, JSON values and row counts", () => {
  expectTypeOf<SQLResponse["meta"][number]["name"]>().toEqualTypeOf<string>();
  expectTypeOf<SQLResponse["data"][number][string]>().toEqualTypeOf<components["schemas"]["JsonValue"]>();
  expectTypeOf<SQLResponse["rows"]>().toEqualTypeOf<number | string>();
  expectTypeOf<SQLResponse["statistics"]["elapsed"]>().toEqualTypeOf<number>();
});

test("query help exposes schema discovery, examples and discovery failures", () => {
  expectTypeOf<QueryHelp["tables"][number]["columns"][number]["type"]>().toEqualTypeOf<string>();
  expectTypeOf<QueryHelp["metadata"]["error"]>().toEqualTypeOf<string | null | undefined>();
  expectTypeOf<QueryHelp["examples"][number]["sql"]>().toEqualTypeOf<string>();
  expectTypeOf<QueryHelp["guide"]>().toEqualTypeOf<string>();
});
