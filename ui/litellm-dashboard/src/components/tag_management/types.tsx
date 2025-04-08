export interface Tag {
  name: string;
  description?: string;
  models: string[];
  created_at: string;
  updated_at: string;
  created_by?: string;
  updated_by?: string;
}

export interface TagInfoRequest {
  names: string[];
}

export interface TagNewRequest {
  name: string;
  description?: string;
  models: string[];
}

export interface TagUpdateRequest {
  name: string;
  description?: string;
  models: string[];
}

export interface TagDeleteRequest {
  name: string;
}

// The API returns a dictionary of tags where the key is the tag name
export type TagListResponse = Record<string, Tag>;
export type TagInfoResponse = Record<string, Tag>; 