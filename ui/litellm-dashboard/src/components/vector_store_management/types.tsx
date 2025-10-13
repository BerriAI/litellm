export interface VectorStore {
  vector_store_id: string;
  custom_llm_provider: string;
  vector_store_name?: string;
  vector_store_description?: string;
  vector_store_metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
  created_by?: string;
  updated_by?: string;
}

export interface VectorStoreInfoRequest {
  vector_store_id: string;
}

export interface VectorStoreNewRequest {
  vector_store_id: string;
  custom_llm_provider: string;
  vector_store_name?: string;
  vector_store_description?: string;
  vector_store_metadata?: Record<string, any>;
}

export interface VectorStoreUpdateRequest {
  vector_store_id: string;
  custom_llm_provider?: string;
  vector_store_name?: string;
  vector_store_description?: string;
  vector_store_metadata?: Record<string, any>;
}

export interface VectorStoreDeleteRequest {
  vector_store_id: string;
}

export interface VectorStoreListResponse {
  object: string;
  data: VectorStore[];
  total_count: number;
  current_page: number;
  total_pages: number;
}
