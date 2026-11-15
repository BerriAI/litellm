import { describe, expectTypeOf, test } from "vitest";

import type { components } from "@/lib/http/schema";
import type { SearchToolInfo, SearchToolLiteLLMParams } from "@/app/(dashboard)/search-tools/_components/types";
import type { SearchToolPayload } from "@/app/(dashboard)/search-tools/_components/searchToolPayload";

describe("search tool types", () => {
  test("litellm params are the generated OpenAPI component", () => {
    expectTypeOf<SearchToolLiteLLMParams>().toEqualTypeOf<components["schemas"]["SearchToolLiteLLMParams"]>();
  });

  test("a param the backend has not declared does not type-check", () => {
    // @ts-expect-error search_engine_id type-checks only once litellm/types/search.py declares it
    const params: SearchToolLiteLLMParams = { search_provider: "google_pse", search_engine_id: "cx-123" };
    expectTypeOf(params).toExtend<{ search_provider: string }>();
  });

  test("tool info carries a description and nothing else", () => {
    // @ts-expect-error search_tool_info has no owner field
    const info: SearchToolInfo = { description: "finds things", owner: "platform-team" };
    expectTypeOf(info).toExtend<{ description?: string | null }>();
  });

  test("the payload sends litellm params the backend declares", () => {
    expectTypeOf<SearchToolPayload["litellm_params"]>().toEqualTypeOf<SearchToolLiteLLMParams>();
  });
});
