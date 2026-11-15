import React, { useEffect, useState } from "react";
import { VectorStore } from "./types";
import { vectorStoreListCall } from "../networking";
import { MultiSelect } from "@/components/shared/MultiSelect";
import i18n from "@/i18n/i18n";
interface VectorStoreSelectorProps {
  id?: string;
  "aria-describedby"?: string;
  onChange: (selectedVectorStores: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const VectorStoreSelector: React.FC<VectorStoreSelectorProps> = ({
  id,
  "aria-describedby": describedBy,
  onChange,
  value,
  className,
  accessToken,
  placeholder = i18n.t("vectorStoreSelector.placeholder"),
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
        id={id}
        aria-describedby={describedBy}
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
