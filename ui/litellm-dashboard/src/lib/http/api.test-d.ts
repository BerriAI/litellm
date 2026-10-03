import { expectTypeOf, test } from "vitest";
import type { components } from "./schema";
import type { getClaudeCodePluginsList, userListCall } from "@/components/networking";
import type { TraceMessage, UIMessage } from "@/components/view_logs/TraceView/traceTypes";
import type { TagNewRequest, TagUpdateRequest } from "@/components/tag_management/types";

test("API read functions expose the generated response contracts", () => {
  expectTypeOf<Awaited<ReturnType<typeof userListCall>>>().toEqualTypeOf<components["schemas"]["UserListResponse"]>();
  expectTypeOf<Awaited<ReturnType<typeof getClaudeCodePluginsList>>>().toEqualTypeOf<
    components["schemas"]["ListPluginsResponse"]
  >();
});

test("trace messages accept API names while adapting tool calls", () => {
  expectTypeOf<TraceMessage["name"]>().toEqualTypeOf<UIMessage["name"]>();
  expectTypeOf<UIMessage["content"]>().toEqualTypeOf<TraceMessage["content"]>();
});

test("tag payloads accept the generated request contracts", () => {
  expectTypeOf<TagNewRequest>().toEqualTypeOf<components["schemas"]["TagNewRequest"]>();
  expectTypeOf<TagUpdateRequest>().toEqualTypeOf<components["schemas"]["TagUpdateRequest"]>();
});
