import type { components } from "@/lib/http/schema";

export interface Tag {
  name: string;
  description?: string;
  models: string[]; // model IDs
  model_info?: { [key: string]: string }; // maps model_id to model_name
  created_at: string;
  updated_at: string;
  created_by?: string;
  updated_by?: string;
  litellm_budget_table?: {
    max_budget?: number;
    soft_budget?: number;
    tpm_limit?: number;
    rpm_limit?: number;
    max_parallel_requests?: number;
    budget_duration?: string;
    model_max_budget?: any;
  };
}

export type TagInfoRequest = components["schemas"]["TagInfoRequest"];
export type TagNewRequest = components["schemas"]["TagNewRequest"];
export type TagUpdateRequest = components["schemas"]["TagUpdateRequest"];
export type TagDeleteRequest = components["schemas"]["TagDeleteRequest"];

// GET /tag/list returns an array; the shape comes from the backend's response_model.
export type TagListItem = components["schemas"]["TagListItem"];
export type TagListResponse = TagListItem[];

// Views built on the hand-written Tag model convert at the boundary. Dynamic tags (seen only in
// spend) come back without models or timestamps, so fill the fields Tag treats as required.
export const toTag = (item: TagListItem): Tag => ({
  name: item.name,
  description: item.description ?? undefined,
  models: item.models ?? [],
  model_info: (item.model_info ?? undefined) as Tag["model_info"],
  created_at: item.created_at ?? "",
  updated_at: item.updated_at ?? "",
  created_by: item.created_by ?? undefined,
  litellm_budget_table: (item.litellm_budget_table ?? undefined) as Tag["litellm_budget_table"],
});
// POST /tag/info returns a dictionary of tags keyed by tag name
export type TagInfoResponse = Record<string, Tag>;
