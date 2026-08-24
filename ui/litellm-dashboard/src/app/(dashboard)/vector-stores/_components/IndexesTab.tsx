"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

import { toast } from "@/lib/toast";
import { indexesListCall } from "@/components/networking";
import { VectorStore } from "@/components/vector_store_management/types";

import IndexesTable from "./IndexesTable";

export interface VectorStoreIndex {
  id: string;
  index_name: string;
  litellm_params: {
    vector_store_index: string;
    vector_store_name: string;
  };
  index_info?: Record<string, unknown> | null;
  created_at?: string | null;
  created_by?: string | null;
  updated_at?: string | null;
  updated_by?: string | null;
}

interface IndexesTabProps {
  accessToken: string | null;
  vectorStores: VectorStore[];
  onViewVectorStore: (vectorStoreId: string) => void;
}

const IndexesTab: React.FC<IndexesTabProps> = ({ accessToken, vectorStores, onViewVectorStore }) => {
  const [indexes, setIndexes] = useState<VectorStoreIndex[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const vectorStoreIdsByName = useMemo(
    () =>
      new Map(
        vectorStores.flatMap((store) =>
          store.vector_store_name ? [[store.vector_store_name, store.vector_store_id] as const] : [],
        ),
      ),
    [vectorStores],
  );

  const resolveVectorStoreId = useCallback((name: string) => vectorStoreIdsByName.get(name), [vectorStoreIdsByName]);

  useEffect(() => {
    const fetchIndexes = async () => {
      if (!accessToken) {
        setIsLoading(false);
        return;
      }
      try {
        const response = await indexesListCall(accessToken);
        setIndexes(response.data || []);
      } catch (error) {
        console.error("Error fetching indexes:", error);
        toast.fromError("Error fetching indexes: " + error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchIndexes();
  }, [accessToken]);

  return (
    <div className="w-full">
      <p className="mb-4 text-sm text-muted-foreground">
        Vector store indexes registered on this proxy via the <code>/v1/indexes</code> API. See the{" "}
        <a
          href="https://docs.litellm.ai/docs/providers/azure_ai/azure_ai_vector_stores_passthrough"
          target="_blank"
          rel="noopener noreferrer"
          className="text-info hover:underline"
        >
          vector store index docs
        </a>{" "}
        for how this works. Index passthrough is supported for Azure AI Search and Milvus today; support for more
        providers can be added, so please{" "}
        <a
          href="https://github.com/BerriAI/litellm/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="text-info hover:underline"
        >
          file a GitHub issue
        </a>{" "}
        if you want your provider supported.
      </p>
      <div className="grid grid-cols-1 gap-2 pt-2 pb-2 w-full">
        <IndexesTable
          data={indexes}
          isLoading={isLoading}
          resolveVectorStoreId={resolveVectorStoreId}
          onViewVectorStore={onViewVectorStore}
        />
      </div>
    </div>
  );
};

export default IndexesTab;
