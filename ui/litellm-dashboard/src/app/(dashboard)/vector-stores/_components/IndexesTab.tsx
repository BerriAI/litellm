"use client";

import React, { useEffect, useState } from "react";

import NotificationsManager from "@/components/molecules/notifications_manager";
import { indexesListCall } from "@/components/networking";

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
}

const IndexesTab: React.FC<IndexesTabProps> = ({ accessToken }) => {
  const [indexes, setIndexes] = useState<VectorStoreIndex[]>([]);
  const [isLoading, setIsLoading] = useState(true);

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
        NotificationsManager.fromBackend("Error fetching indexes: " + error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchIndexes();
  }, [accessToken]);

  return (
    <div className="w-full">
      <p className="mb-4 text-sm text-muted-foreground">
        Vector store indexes registered on this proxy via the <code>/v1/indexes</code> API.
      </p>
      <div className="grid grid-cols-1 gap-2 pt-2 pb-2 w-full">
        <IndexesTable data={indexes} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default IndexesTab;
