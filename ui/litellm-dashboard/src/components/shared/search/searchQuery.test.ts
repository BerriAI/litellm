import { describe, expect, it } from "vitest";

import { NOTE_QUERY } from "./__fixtures__/notes";
import { parseQuery } from "./language";
import { toSearchQuery } from "./searchQuery";

const query = (text: string) => toSearchQuery(parseQuery(NOTE_QUERY, text));

describe("toSearchQuery", () => {
  it("turns clauses into terms and typed filters, choosing the op from negation and wildcards", () => {
    expect(query('refund "vector stores" tag:cron -tag:*bill* title:*lead -id:aaa111')).toEqual({
      text: ["refund", "vector stores"],
      filters: [
        { field: "tag", op: "eq", value: "cron" },
        { field: "tag", op: "nglob", value: "*bill*" },
        { field: "title", op: "glob", value: "*lead" },
        { field: "id", op: "neq", value: "aaa111" },
      ],
    });
  });

  it("drops a key still waiting for its value and empty quoted terms, so typing never blanks the list", () => {
    expect(query('tag: -title: ""')).toEqual({ text: [], filters: [] });
    expect(query("refund tag:")).toEqual({ text: ["refund"], filters: [] });
  });

  it("keeps an unknown key as free text", () => {
    expect(query("foo:bar")).toEqual({ text: ["foo:bar"], filters: [] });
  });
});
