import { zodFunction } from "openai/helpers/zod";
import { z } from "zod";
import { fetchDocsIndex } from "@/lib/http/docsIndex";
import { PAGE_DOCS } from "@/lib/pageDocs";

export { DOCS_INDEX_URL } from "@/lib/http/docsIndex";

export interface DocEntry {
  readonly title: string;
  readonly url: string;
  readonly description: string;
}

const ENTRY_PATTERN = /^- \[([^\]]+)\]\(([^)\s]+)\)\s*(?::\s*(.*))?$/;

const normalizeDocsUrl = (rawUrl: string): string | null => {
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "https:" || parsed.host !== "docs.litellm.ai") return null;
  } catch {
    return null;
  }
  const noMarkdown = rawUrl.replace(/\.md$/, "");
  if (!noMarkdown.endsWith("/index")) return noMarkdown;
  const parent = noMarkdown.slice(0, -"/index".length);
  return parent === "https://docs.litellm.ai/docs" ? `${parent}/` : parent;
};

export function parseDocsIndex(text: string): readonly DocEntry[] {
  return text.split("\n").flatMap((line) => {
    const match = ENTRY_PATTERN.exec(line.trim());
    const url = match ? normalizeDocsUrl(match[2]) : null;
    if (!match || url === null) return [];
    return [{ title: match[1], url, description: match[3] ?? "" }];
  });
}

const STOPWORDS: ReadonlySet<string> = new Set([
  "how",
  "do",
  "i",
  "the",
  "to",
  "a",
  "an",
  "set",
  "up",
  "in",
  "on",
  "for",
  "of",
  "and",
  "is",
  "what",
  "my",
  "can",
  "with",
]);

const queryTokens = (query: string): readonly string[] => [
  ...new Set(
    query
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((token) => token.length >= 2 && !STOPWORDS.has(token)),
  ),
];

const tokenForms = (token: string): readonly string[] =>
  token.endsWith("s") && token.length > 2 ? [token, token.slice(0, -1)] : [token];

export function searchDocs(entries: readonly DocEntry[], query: string, limit = 5): readonly DocEntry[] {
  const tokens = queryTokens(query);
  if (tokens.length === 0) return [];
  const scored = entries.flatMap((entry, index) => {
    const title = entry.title.toLowerCase();
    const path = entry.url.toLowerCase();
    const description = entry.description.toLowerCase();
    const score = tokens.reduce((total, token) => {
      const forms = tokenForms(token);
      return (
        total +
        (forms.some((form) => title.includes(form)) ? 3 : 0) +
        (forms.some((form) => path.includes(form)) ? 2 : 0) +
        (forms.some((form) => description.includes(form)) ? 1 : 0)
      );
    }, 0);
    return score > 0 ? [{ entry, score, index }] : [];
  });
  return scored
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, limit)
    .map(({ entry }) => entry);
}

const bundledEntries: readonly DocEntry[] = [
  ...new Map(
    [...PAGE_DOCS.values()]
      .flat()
      .map((topic) => [topic.href, { title: topic.title, url: topic.href, description: "" }]),
  ).values(),
];

let docsIndexPromise: Promise<readonly DocEntry[]> | null = null;

function loadDocsIndex(fetchIndex: () => Promise<string>): Promise<readonly DocEntry[]> {
  docsIndexPromise ??= fetchIndex()
    .then(parseDocsIndex)
    .catch((error: unknown) => {
      docsIndexPromise = null;
      throw error;
    });
  return docsIndexPromise;
}

export interface DocsSearchToolOptions {
  readonly signal: AbortSignal;
  readonly beforeTool: () => void;
  readonly fetchIndex?: (signal: AbortSignal) => Promise<string>;
}

const truncate = (text: string): string => (text.length > 200 ? `${text.slice(0, 200)}…` : text);

const loadCorpus = async (
  options: DocsSearchToolOptions,
  fetchIndex: (signal: AbortSignal) => Promise<string>,
): Promise<{ entries: readonly DocEntry[]; source: "docs" | "bundled" }> => {
  try {
    const fetched = await loadDocsIndex(() => fetchIndex(options.signal));
    return {
      entries: [...fetched, ...bundledEntries.filter((b) => !fetched.some((f) => f.url === b.url))],
      source: "docs",
    };
  } catch {
    return { entries: bundledEntries, source: "bundled" };
  }
};

export async function runDocsSearch(options: DocsSearchToolOptions, query: string) {
  options.beforeTool();
  const { entries, source } = await loadCorpus(options, options.fetchIndex ?? fetchDocsIndex);
  return {
    operation: "docs_search",
    success: true,
    source,
    results: searchDocs(entries, query).map((entry) => ({
      title: entry.title,
      url: entry.url,
      description: truncate(entry.description),
    })),
  };
}

export function createDocsSearchTool(options: DocsSearchToolOptions) {
  const definition = {
    name: "docs_search",
    description:
      "Search the LiteLLM documentation for setup, configuration and concept questions. Returns page titles, links and summaries. Read-only.",
    parameters: z.strictObject({ query: z.string().min(1).max(200) }),
    function: (rawArgs: unknown) => runDocsSearch(options, (rawArgs as { query: string }).query),
  };
  return zodFunction(definition);
}
