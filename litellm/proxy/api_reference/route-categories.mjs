export const categories = [
  {
    id: "ai",
    label: "AI Routes",
    description: "Models, completions, media, and provider APIs.",
  },
  {
    id: "management",
    label: "Management Routes",
    description: "Keys, teams, budgets, and gateway settings.",
  },
  {
    id: "mcp",
    label: "MCP Routes",
    description: "Servers, tools, discovery, and MCP connections.",
  },
];

const methods = new Set([
  "get",
  "put",
  "post",
  "delete",
  "options",
  "head",
  "patch",
  "trace",
]);

const aiTags = new Set([
  "chat/completions",
  "completions",
  "responses",
  "embeddings",
  "moderations",
  "audio",
  "assistants",
  "llm utils",
  "rerank",
  "ocr",
  "rag",
  "videos",
  "containers",
  "search",
  "images",
  "fine-tuning",
  "batch",
  "files",
  "google genai endpoints",
  "interactions",
  "vector_stores",
  "realtime",
  "websocket",
  "llm_passthrough",
  "openai pass-through",
  "pass-through",
  "anthropic_passthrough",
  "anthropic_skills",
  "evals",
  "decisions",
  "a2a",
]);

const preferredTags = [
  "chat/completions",
  "completions",
  "responses",
  "embeddings",
  "images",
  "audio",
  "models",
  "model management",
  "key management",
  "team management",
  "spend",
  "mcp_management",
  "mcp_rest",
  "mcp_discoverable",
];

export function categoryForOperation(path, operation, method) {
  const tags = (operation.tags || []).map((tag) => tag.toLowerCase());
  if (
    tags.some((tag) => /\bmcp(?:\b|_)/.test(tag)) ||
    path.split("/").some((segment) => /^mcp(?:$|[-_])/.test(segment))
  )
    return "mcp";

  if (
    (method === "get" && /^\/(?:v1\/)?models(?:\/\{model_id\})?$/.test(path)) ||
    tags.some((tag) => aiTags.has(tag))
  )
    return "ai";

  return "management";
}

function scopedPaths(paths, category) {
  return Object.fromEntries(
    Object.entries(paths).flatMap(([path, item]) => {
      const entries = Object.entries(item);
      const operations = entries.filter(
        ([method, operation]) =>
          methods.has(method) &&
          categoryForOperation(path, operation, method) === category,
      );
      if (!operations.length) {
        return category === "management" && item.$ref ? [[path, item]] : [];
      }
      return [
        [
          path,
          Object.fromEntries([
            ...entries.filter(([key]) => !methods.has(key)),
            ...operations,
          ]),
        ],
      ];
    }),
  );
}

function scopedTags(schema, paths) {
  const names = [
    ...new Set(
      Object.values(paths).flatMap((item) =>
        Object.entries(item).flatMap(([method, operation]) =>
          methods.has(method) ? operation.tags || [] : [],
        ),
      ),
    ),
  ];
  const metadata = new Map((schema.tags || []).map((tag) => [tag.name, tag]));
  const rank = (name) => {
    const index = preferredTags.indexOf(name.toLowerCase());
    return index < 0 ? preferredTags.length : index;
  };
  return names
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((name) => metadata.get(name) || { name });
}

export function groupSchema(schema) {
  return categories.map((category) => {
    const paths = scopedPaths(schema.paths || {}, category.id);
    const count = Object.values(paths).reduce(
      (total, item) =>
        total + Object.keys(item).filter((key) => methods.has(key)).length,
      0,
    );
    return {
      ...category,
      count,
      schema: { ...schema, paths, tags: scopedTags(schema, paths) },
    };
  });
}
