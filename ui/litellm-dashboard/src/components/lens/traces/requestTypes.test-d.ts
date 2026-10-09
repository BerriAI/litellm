import { expectTypeOf, test } from "vitest";
import type { SpanErrorQuery, SpanQuery, TraceDetailQuery, TraceListQuery, TraceQueryBody } from "./types";

test("trace request aliases match the generated OpenAPI shapes", () => {
  expectTypeOf<TraceListQuery>().toEqualTypeOf<{
    start_ms?: number | null;
    end_ms?: number | null;
    cursor?: string | null;
  }>();
  expectTypeOf<TraceDetailQuery>().toEqualTypeOf<{
    trace_ref?: string;
    cursor?: string | null;
    page_size?: number | null;
  }>();
  expectTypeOf<SpanQuery>().toEqualTypeOf<{ trace_ref?: string }>();
  expectTypeOf<SpanErrorQuery>().toEqualTypeOf<{
    trace_ref?: string;
    cursor?: string | null;
  }>();
  expectTypeOf<TraceQueryBody>().toEqualTypeOf<{ sql: string }>();
});
