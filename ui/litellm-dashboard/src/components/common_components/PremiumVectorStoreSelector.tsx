import React from "react";
import { Text } from "@tremor/react";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";

interface PremiumVectorStoreSelectorProps {
  onChange: (values: string[]) => void;
  value: string[];
  accessToken: string;
  placeholder?: string;
  premiumUser?: boolean;
}

export function PremiumVectorStoreSelector({
  onChange,
  value,
  accessToken,
  placeholder = "Select vector stores",
  premiumUser = false,
}: PremiumVectorStoreSelectorProps) {
  if (!premiumUser) {
    return (
      <div>
        <div className="flex flex-wrap gap-2 mb-3">
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium opacity-50">
            ✨ premium-vector-store-1
          </div>
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium opacity-50">
            ✨ premium-vector-store-2
          </div>
        </div>
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <Text className="text-sm text-yellow-800">
            Vector store access control is a LiteLLM Enterprise feature. Get a trial key{" "}
            <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
              here
            </a>
            .
          </Text>
        </div>
      </div>
    );
  }

  return <VectorStoreSelector onChange={onChange} value={value} accessToken={accessToken} placeholder={placeholder} />;
}

export default PremiumVectorStoreSelector;
