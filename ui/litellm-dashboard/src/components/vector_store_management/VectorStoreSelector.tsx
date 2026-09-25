import React, { useEffect, useState } from "react";
import { VectorStore } from "./types";
import { vectorStoreListCall } from "../networking";
import { MultiSelect } from "@/components/shared/MultiSelect";
interface VectorStoreSelectorProps {
  onChange: (selectedVectorStores: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const VectorStoreSelector: React.FC<VectorStoreSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select vector stores",
  disabled = false,
}) => {
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStores(response.data);
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchVectorStores();
  }, [accessToken]);

  return (
    <div className="min-w-0">
      <MultiSelect
        placeholder={placeholder}
        onValueChange={onChange}
        value={value}
        loading={loading}
        className={className}
        disabled={disabled}
        options={vectorStores.map((store) => ({
          label: `${store.vector_store_name || store.vector_store_id} (${store.vector_store_id})`,
          value: store.vector_store_id,
          description: store.vector_store_description || undefined,
        }))}
      />
    </div>
  );
};

export default VectorStoreSelector;
