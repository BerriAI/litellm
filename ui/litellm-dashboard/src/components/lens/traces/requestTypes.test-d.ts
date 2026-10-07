import { expectTypeOf, test } from "vitest";
import type {
  SpanErrorQuery,
  SpanQuery,
  TraceAgentList,
  TraceAgentsQuery,
  TraceDetailQuery,
  TraceListQuery,
  TraceQueryBody,
} from "./types";

test("trace request aliases match the generated OpenAPI shapes", () => {
  expectTypeOf<TraceListQuery>().toEqualTypeOf<{
    start_ms?: number | null;
    end_ms?: number | null;
    cursor?: string | null;
    agent?: string | null;
  }>();
  expectTypeOf<TraceAgentsQuery>().toEqualTypeOf<{
    start_ms?: number | null;
    end_ms?: number | null;
  }>();
  expectTypeOf<TraceAgentList>().toEqualTypeOf<{ data: string[] }>();
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
