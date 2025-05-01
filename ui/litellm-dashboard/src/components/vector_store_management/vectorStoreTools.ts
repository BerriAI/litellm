/**
 * Utility functions for creating vector store tool configurations
 */

/**
 * Creates a tools array with vector store IDs for use with LLM API calls
 * 
 * @param vectorStoreIds - Array of vector store IDs
 * @returns The properly formatted tools array for LLM API calls
 */
export function createVectorStoreTools(vectorStoreIds?: string[]) {
  if (!vectorStoreIds || vectorStoreIds.length === 0) {
    return undefined;
  }

  return [
    {
      type: "file_search",
      vector_store_ids: vectorStoreIds
    }
  ];
} 