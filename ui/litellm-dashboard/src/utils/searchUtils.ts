type SearchField = string | null | undefined;

const normalizeTerm = (term: string): string => term.trim().toLowerCase();

export function matchesSearchTerm(term: string, fields: ReadonlyArray<SearchField>): boolean {
  const needle = normalizeTerm(term);
  if (needle === "") return true;

  const haystacks = fields.filter((field): field is string => typeof field === "string").map((f) => f.toLowerCase());
  if (haystacks.some((haystack) => haystack.includes(needle))) return true;

  return needle.split(/\s+/).every((word) => haystacks.some((haystack) => haystack.includes(word)));
}

export function filterBySearchTerm<T>(
  items: ReadonlyArray<T>,
  term: string,
  fields: (item: T) => ReadonlyArray<SearchField>,
): T[] {
  return items.filter((item) => matchesSearchTerm(term, fields(item)));
}

export function rankBySearchRelevance<T>(items: ReadonlyArray<T>, term: string, name: (item: T) => string): T[] {
  const needle = normalizeTerm(term);
  if (needle === "") return [...items];

  const score = (item: T): number => {
    const candidate = name(item).toLowerCase();
    return (candidate === needle ? 1000 : 0) + (candidate.startsWith(needle) ? 100 : 0) + (1000 - candidate.length);
  };
  return [...items].sort((a, b) => score(b) - score(a));
}
