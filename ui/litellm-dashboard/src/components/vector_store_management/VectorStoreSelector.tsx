import React, { useEffect, useState } from 'react';
import { Select, Typography, Tooltip } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import { VectorStore } from './types';
import { vectorStoreListCall } from '../networking';
interface VectorStoreSelectorProps {
  onChange: (selectedVectorStores: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
}

const VectorStoreSelector: React.FC<VectorStoreSelectorProps> = ({ 
  onChange, 
  value, 
  className, 
  accessToken 
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
    <div>
      <Select
        mode="multiple"
        placeholder="Select vector stores"
        onChange={onChange}
        value={value}
        loading={loading}
        className={className}
        options={vectorStores.map(store => ({
          label: `${store.vector_store_name || store.vector_store_id} (${store.vector_store_id})`,
          value: store.vector_store_id,
          title: store.vector_store_description || store.vector_store_id,
        }))}
        optionFilterProp="label"
        showSearch
        style={{ width: '100%' }}
      />
    </div>
  );
};

export default VectorStoreSelector; 