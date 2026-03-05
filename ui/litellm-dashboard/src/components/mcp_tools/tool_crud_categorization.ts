export type CRUDCategory = "Create" | "Read" | "Update" | "Delete" | "Other";

interface ToolCategoryKeywords {
  Create: string[];
  Read: string[];
  Update: string[];
  Delete: string[];
}

const CRUD_KEYWORDS: ToolCategoryKeywords = {
  Create: [
    "create",
    "add",
    "insert",
    "new",
    "post",
    "register",
    "init",
    "initialize",
    "setup",
    "make",
    "generate",
    "build",
    "construct",
  ],
  Read: [
    "get",
    "fetch",
    "list",
    "search",
    "find",
    "read",
    "query",
    "view",
    "retrieve",
    "show",
    "describe",
    "check",
    "load",
    "export",
    "download",
    "info",
  ],
  Update: [
    "update",
    "edit",
    "modify",
    "patch",
    "change",
    "alter",
    "replace",
    "rename",
  ],
  Delete: [
    "delete",
    "remove",
    "drop",
    "clear",
    "destroy",
    "purge",
    "revoke",
    "cancel",
    "unregister",
  ],
};

function splitCamelCase(text: string): string {
  return text.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/([A-Z])([A-Z][a-z])/g, "$1 $2");
}

export function categorizeTool(toolName: string, description?: string): CRUDCategory {
  const normalizedName = splitCamelCase(toolName).toLowerCase();
  const searchText = `${normalizedName} ${description || ""}`.toLowerCase();

  for (const [category, keywords] of Object.entries(CRUD_KEYWORDS)) {
    for (const keyword of keywords) {
      const pattern = new RegExp(`\\b${keyword}`, "i");
      if (pattern.test(searchText)) {
        return category as CRUDCategory;
      }
    }
  }

  return "Other";
}

export interface CategorizedTool {
  name: string;
  description?: string;
  category: CRUDCategory;
}

export function categorizeTools(tools: Array<{ name: string; description?: string }>): CategorizedTool[] {
  return tools.map((tool) => ({
    ...tool,
    category: categorizeTool(tool.name, tool.description),
  }));
}

export function groupToolsByCategory(tools: CategorizedTool[]): Record<CRUDCategory, CategorizedTool[]> {
  const grouped: Record<CRUDCategory, CategorizedTool[]> = {
    Create: [],
    Read: [],
    Update: [],
    Delete: [],
    Other: [],
  };

  tools.forEach((tool) => {
    grouped[tool.category].push(tool);
  });

  return grouped;
}

export const CRUD_CATEGORY_ORDER: CRUDCategory[] = ["Create", "Read", "Update", "Delete", "Other"];

export const CRUD_CATEGORY_COLORS: Record<CRUDCategory, { bg: string; border: string; text: string; badge: string }> = {
  Create: {
    bg: "bg-green-50",
    border: "border-green-300",
    text: "text-green-700",
    badge: "bg-green-100 text-green-800",
  },
  Read: {
    bg: "bg-blue-50",
    border: "border-blue-300",
    text: "text-blue-700",
    badge: "bg-blue-100 text-blue-800",
  },
  Update: {
    bg: "bg-yellow-50",
    border: "border-yellow-300",
    text: "text-yellow-700",
    badge: "bg-yellow-100 text-yellow-800",
  },
  Delete: {
    bg: "bg-red-50",
    border: "border-red-300",
    text: "text-red-700",
    badge: "bg-red-100 text-red-800",
  },
  Other: {
    bg: "bg-gray-50",
    border: "border-gray-300",
    text: "text-gray-700",
    badge: "bg-gray-100 text-gray-800",
  },
};
