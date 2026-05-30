// PROTOTYPE: localStorage-backed MCP Server *Templates* store.
// A template captures everything you'd put into the Add New MCP Server modal
// EXCEPT values for variables — it only declares which variables exist and
// whether each one is `instance`-scoped (admin fills it per-instance) or
// `per_user` (each end user supplies their own via the Variables tab).
//
// This file is intentionally throw-away — the prototype's goal is to show the
// flow to the customer, not to be production code.

export type TemplateVariableScope = "instance" | "per_user";

export interface TemplateVariable {
  name: string;
  scope: TemplateVariableScope;
}

export interface MCPTemplate {
  template_id: string;
  name: string;
  description?: string;
  transport: string; // "http" | "sse" | "stdio" | "openapi"
  url?: string;
  auth_type?: string;
  variables: TemplateVariable[];
  static_headers?: Array<{ header: string; value: string }>;
  logo_url?: string;
  created_at: string;
  // PROTOTYPE: snapshot of every other form field (OAuth URLs, stdio config,
  // BYOK toggles, access groups, allowed_tools, cost config, …) so editing
  // a template restores the full Add-MCP-Server form, and creating an
  // instance from this template starts from the same baseline.
  form_snapshot?: Record<string, any>;
}

const KEY = "mock-mcp-templates::v1";
const CHANGE_EVENT = "mock-mcp-templates-changed";

export function listTemplates(): MCPTemplate[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function getTemplate(id: string): MCPTemplate | null {
  return listTemplates().find((t) => t.template_id === id) ?? null;
}

export function saveTemplate(template: MCPTemplate): void {
  if (typeof window === "undefined") return;
  const all = listTemplates();
  const idx = all.findIndex((t) => t.template_id === template.template_id);
  if (idx >= 0) all[idx] = template;
  else all.push(template);
  window.localStorage.setItem(KEY, JSON.stringify(all));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

export function deleteTemplate(id: string): void {
  if (typeof window === "undefined") return;
  const all = listTemplates().filter((t) => t.template_id !== id);
  window.localStorage.setItem(KEY, JSON.stringify(all));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

export function subscribeTemplatesChanged(handler: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(CHANGE_EVENT, handler);
  const storageHandler = (e: StorageEvent) => {
    if (e.key === KEY) handler();
  };
  window.addEventListener("storage", storageHandler);
  return () => {
    window.removeEventListener(CHANGE_EVENT, handler);
    window.removeEventListener("storage", storageHandler);
  };
}

export function newTemplateId(): string {
  return `tpl_${Math.random().toString(36).slice(2, 10)}`;
}
