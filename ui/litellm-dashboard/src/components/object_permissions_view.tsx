import React, { useState, useEffect } from "react";
import { Card, Text, Badge } from "@tremor/react";
import { ServerIcon, DatabaseIcon } from "@heroicons/react/outline";
import { vectorStoreListCall } from "./networking";

interface VectorStoreDetails {
  vector_store_id: string;
  vector_store_name?: string;
}

interface ObjectPermission {
  object_permission_id: string;
  mcp_servers: string[];
  vector_stores: string[];
}

interface ObjectPermissionsViewProps {
  objectPermission?: ObjectPermission;
  variant?: "card" | "inline";
  className?: string;
  accessToken?: string | null;
  premiumUser?: boolean;
}

export function ObjectPermissionsView({ 
  objectPermission, 
  variant = "card",
  className = "",
  accessToken,
  premiumUser = false
}: ObjectPermissionsViewProps) {
  const vectorStores = objectPermission?.vector_stores || [];
  const mcpServers = objectPermission?.mcp_servers || [];
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStoreDetails[]>([]);

  // Premium user check
  if (!premiumUser) {
    const mockContent = (
      <div className={variant === "card" ? "grid grid-cols-1 md:grid-cols-2 gap-6" : "space-y-4"}>
        {/* Mock Vector Stores Section */}
        <div className="space-y-3 opacity-50">
          <div className="flex items-center gap-2">
            <DatabaseIcon className="h-4 w-4 text-blue-600" />
            <Text className="font-semibold text-gray-900">✨ Vector Stores</Text>
            <Badge color="blue" size="xs">
              2
            </Badge>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium">
              ✨ premium-vector-store-1
            </div>
            <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium">
              ✨ premium-vector-store-2
            </div>
          </div>
        </div>
      </div>
    );

    if (variant === "card") {
      return (
        <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
          <div className="flex items-center gap-2 mb-6">
            <div>
              <Text className="font-semibold text-gray-900">✨ Object Permissions</Text>
              <Text className="text-xs text-gray-500">
                Access control for Vector Stores
              </Text>
            </div>
          </div>
          {mockContent}
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <Text className="text-sm text-yellow-800">
              This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer" className="underline">here</a>.
            </Text>
          </div>
        </div>
      );
    }

    return (
      <div className={`${className}`}>
        <Text className="font-medium text-gray-900 mb-3">✨ Object Permissions</Text>
        {mockContent}
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <Text className="text-sm text-yellow-800">
            This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer" className="underline">here</a>.
          </Text>
        </div>
      </div>
    );
  }

  // Fetch vector store details when component mounts
  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken || vectorStores.length === 0) return;
      
      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStoreDetails(response.data.map((store: any) => ({
            vector_store_id: store.vector_store_id,
            vector_store_name: store.vector_store_name
          })));
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      }
    };

    fetchVectorStores();
  }, [accessToken, vectorStores.length]);

  // Function to get display name for vector store
  const getVectorStoreDisplayName = (storeId: string) => {
    const storeDetail = vectorStoreDetails.find(store => store.vector_store_id === storeId);
    if (storeDetail) {
      return `${storeDetail.vector_store_name || storeDetail.vector_store_id} (${storeDetail.vector_store_id})`;
    }
    return storeId;
  };

  const content = (
    <div className={variant === "card" ? "grid grid-cols-1 md:grid-cols-2 gap-6" : "space-y-4"}>
      {/* Vector Stores Section */}
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

      {/* MCP Servers Section */}
      {/* <div className="space-y-3">
        <div className="flex items-center gap-2">
          <ServerIcon className="h-4 w-4 text-blue-600" />
          <Text className="font-semibold text-gray-900">MCP Servers</Text>
          <Badge color="blue" size="xs">
            {mcpServers.length}
          </Badge>
        </div>
        
        {mcpServers.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {mcpServers.map((server, index) => (
              <div
                key={index}
                className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium"
              >
                {server}
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
            <ServerIcon className="h-4 w-4 text-gray-400" />
            <Text className="text-gray-500 text-sm">No MCP servers configured</Text>
          </div>
        )}
      </div> */}
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <Text className="font-semibold text-gray-900">Object Permissions</Text>
            <Text className="text-xs text-gray-500">
              Access control for Vector Stores
            </Text>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <Text className="font-medium text-gray-900 mb-3">Object Permissions</Text>
      {content}
    </div>
  );
}

export default ObjectPermissionsView; 