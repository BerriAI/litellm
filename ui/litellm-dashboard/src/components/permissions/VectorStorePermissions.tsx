import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Database } from "lucide-react";
import { vectorStoreListCall } from "../networking";

interface VectorStoreDetails {
  vector_store_id: string;
  vector_store_name?: string;
}

interface VectorStorePermissionsProps {
  vectorStores: string[];
  accessToken?: string | null;
}

export function VectorStorePermissions({
  vectorStores,
  accessToken,
}: VectorStorePermissionsProps) {
  const [vectorStoreDetails, setVectorStoreDetails] = useState<
    VectorStoreDetails[]
  >([]);

  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken || vectorStores.length === 0) return;

      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStoreDetails(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            response.data.map((store: any) => ({
              vector_store_id: store.vector_store_id,
              vector_store_name: store.vector_store_name,
            })),
          );
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      }
    };

    fetchVectorStores();
  }, [accessToken, vectorStores.length]);

  const getVectorStoreDisplayName = (storeId: string) => {
    const storeDetail = vectorStoreDetails.find(
      (store) => store.vector_store_id === storeId,
    );
    if (storeDetail) {
      return `${storeDetail.vector_store_name || storeDetail.vector_store_id} (${storeDetail.vector_store_id})`;
    }
    return storeId;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Database className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="font-semibold text-foreground">Vector Stores</span>
        <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 text-xs">
          {vectorStores.length}
        </Badge>
      </div>

      {vectorStores.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {vectorStores.map((store, index) => (
            <div
              key={index}
              className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200 text-sm font-medium"
            >
              {getVectorStoreDisplayName(store)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
          <Database className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground text-sm">
            No vector stores configured
          </span>
        </div>
      )}
    </div>
  );
}

export default VectorStorePermissions;
