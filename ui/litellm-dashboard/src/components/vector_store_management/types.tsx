export interface IngestedFile {
  file_id?: string;
  filename?: string;
  file_url?: string;
  ingested_at: string;
  file_size?: number;
  content_type?: string;
}

export interface VectorStoreMetadata {
  ingested_files?: IngestedFile[];
  [key: string]: any;
}

export interface VectorStore {
  vector_store_id: string;
  custom_llm_provider: string;
  vector_store_name?: string;
  vector_store_description?: string;
  vector_store_metadata?: VectorStoreMetadata;
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

// Document ingestion types
export interface DocumentUpload {
  uid: string;
  name: string;
  status: "uploading" | "done" | "error" | "removed";
  size?: number;
  type?: string;
  originFileObj?: File;
}

export interface RAGIngestRequest {
  file_url?: string;
  file_id?: string;
  ingest_options: {
    vector_store: {
      custom_llm_provider: string;
      vector_store_id?: string;
    };
  };
}

export interface RAGIngestResponse {
  id: string;
  status: "completed" | "processing" | "failed";
  vector_store_id: string;
  file_id: string;
  error?: string;
}
