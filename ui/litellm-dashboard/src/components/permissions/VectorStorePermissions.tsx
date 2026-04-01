import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { DatabaseIcon } from "@heroicons/react/outline";
import { vectorStoreListCall } from "../networking";

interface VectorStoreDetails {
  vector_store_id: string;
  vector_store_name?: string;
}

interface VectorStorePermissionsProps {
  vectorStores: string[];
  accessToken?: string | null;
}

export function VectorStorePermissions({ vectorStores, accessToken }: VectorStorePermissionsProps) {
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStoreDetails[]>([]);

  // Fetch vector store details when component mounts
  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken || vectorStores.length === 0) return;

      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStoreDetails(
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

  // Function to get display name for vector store
  const getVectorStoreDisplayName = (storeId: string) => {
    const storeDetail = vectorStoreDetails.find((store) => store.vector_store_id === storeId);
    if (storeDetail) {
      return `${storeDetail.vector_store_name || storeDetail.vector_store_id} (${storeDetail.vector_store_id})`;
    }
    return storeId;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <DatabaseIcon className="h-4 w-4 text-blue-600" />
        <Text className="font-semibold text-gray-900">Vector Stores</Text>
        <Badge color="blue" size="xs">
          {vectorStores.length}
        </Badge>
      </div>

      {vectorStores.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {vectorStores.map((store, index) => (
            <div
              key={index}
              className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium"
            >
              {getVectorStoreDisplayName(store)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <DatabaseIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No vector stores configured</Text>
        </div>
      )}
    </div>
  );
}

export default VectorStorePermissions;
