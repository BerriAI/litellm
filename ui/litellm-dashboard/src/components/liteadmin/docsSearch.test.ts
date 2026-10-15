import { describe, expect, it, vi } from "vitest";
import {
  createDocsSearchTool,
  parseDocsIndex,
  runDocsSearch,
  searchDocs,
  type DocEntry,
  type DocsSearchToolOptions,
} from "./docsSearch";

const INDEX_FIXTURE = `# LiteLLM

Some prose that is not an entry.

- [Setting Team Budgets](https://docs.litellm.ai/docs/proxy/team_budgets.md): Give each team a spend budget
- [How Model Access Works](https://docs.litellm.ai/docs/proxy/model_access_guide.md)
- [Vector stores](https://docs.litellm.ai/docs/vector_stores/index.md): Store and search embeddings
- [Docs home](https://docs.litellm.ai/docs/index.md): Documentation index
- [Blog](https://litellm.ai/blog): Not on the docs site
not an entry at all
`;

describe("parseDocsIndex", () => {
  it("parses entries, unwraps index pages, and skips non-docs links and non-entry lines", () => {
    const entries = parseDocsIndex(INDEX_FIXTURE);
    expect(entries).toEqual([
      {
        title: "Setting Team Budgets",
        url: "https://docs.litellm.ai/docs/proxy/team_budgets",
        description: "Give each team a spend budget",
      },
      {
        title: "How Model Access Works",
        url: "https://docs.litellm.ai/docs/proxy/model_access_guide",
        description: "",
      },
      {
        title: "Vector stores",
        url: "https://docs.litellm.ai/docs/vector_stores",
        description: "Store and search embeddings",
      },
      { title: "Docs home", url: "https://docs.litellm.ai/docs/", description: "Documentation index" },
    ]);
  });
});

describe("searchDocs", () => {
  const entries: readonly DocEntry[] = parseDocsIndex(INDEX_FIXTURE);

  it("ranks the entry whose title matches the query first", () => {
    const corpus: readonly DocEntry[] = [
      { title: "Overview", url: "https://docs.litellm.ai/docs/proxy/team_budgets", description: "team budgets" },
      ...entries,
    ];
    const results = searchDocs(corpus, "team budget");
    expect(results[0]?.title).toBe("Setting Team Budgets");
  });

  it("matches a plural query token against the singular in titles", () => {
    const corpus: readonly DocEntry[] = [
      { title: "Set a budget", url: "https://docs.litellm.ai/docs/a", description: "" },
      { title: "Unrelated", url: "https://docs.litellm.ai/docs/b", description: "" },
    ];
    expect(searchDocs(corpus, "budgets").map((entry) => entry.title)).toEqual(["Set a budget"]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(searchDocs(entries, "zzzz qqqq")).toEqual([]);
  });
});

const invoke = async (options: DocsSearchToolOptions, query: string) => runDocsSearch(options, query);

describe("docs_search tool", () => {
  const signal = new AbortController().signal;

  const freshSearch = async (options: DocsSearchToolOptions, query: string) => {
    vi.resetModules();
    const { runDocsSearch: run } = await import("./docsSearch");
    return run(options, query);
  };

  it("creates a docs_search function tool for the model", () => {
    const tool = createDocsSearchTool({ signal, beforeTool: () => {} });
    expect(tool.type).toBe("function");
    expect(tool.function.name).toBe("docs_search");
    expect(tool.function.strict).toBe(true);
  });

  it("searches the fetched index and reports the docs source", async () => {
    const result = await freshSearch(
      {
        signal,
        beforeTool: () => {},
        fetchIndex: async () => INDEX_FIXTURE,
      },
      "team budget",
    );
    expect(result.source).toBe("docs");
    expect(result.results[0]?.title).toBe("Setting Team Budgets");
  });

  it("falls back to bundled topics when the index fetch fails, then retries on the next call", async () => {
    const fetchIndex = vi.fn(async (_signal: AbortSignal) => {
      throw new Error("offline");
    });
    vi.resetModules();
    const { runDocsSearch: run } = await import("./docsSearch");
    const search = (query: string) => run({ signal, beforeTool: () => {}, fetchIndex }, query);
    const offline = await search("budgets");
    expect(offline.source).toBe("bundled");
    expect(offline.results.length).toBeGreaterThan(0);

    fetchIndex.mockImplementation(async () => INDEX_FIXTURE);
    const retried = await search("team budget");
    expect(fetchIndex).toHaveBeenCalledTimes(2);
    expect(retried.source).toBe("docs");
  });

  it("calls beforeTool so the search counts toward the action limit", async () => {
    const beforeTool = vi.fn();
    await invoke({ signal, beforeTool, fetchIndex: async () => INDEX_FIXTURE }, "caching");
    expect(beforeTool).toHaveBeenCalledTimes(1);
  });
});
