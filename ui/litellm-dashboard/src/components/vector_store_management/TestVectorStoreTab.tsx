import React, { useState } from "react";
import { Card, Select, Typography } from "antd";
import { VectorStoreTester } from "./VectorStoreTester";
import { VectorStore } from "./types";

const { Text, Title } = Typography;

interface TestVectorStoreTabProps {
  accessToken: string | null;
  vectorStores: VectorStore[];
}

const TestVectorStoreTab: React.FC<TestVectorStoreTabProps> = ({ accessToken, vectorStores }) => {
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | undefined>(
    vectorStores.length > 0 ? vectorStores[0].vector_store_id : undefined
  );

  if (!accessToken) {
    return (
      <Card>
        <Text type="secondary">Access token is required to test vector stores.</Text>
      </Card>
    );
  }

  if (vectorStores.length === 0) {
    return (
      <Card>
        <div className="text-center py-8">
          <Text type="secondary">No vector stores available. Create one first to test it.</Text>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="space-y-4">
          <div>
            <Title level={5}>Select Vector Store</Title>
            <Text type="secondary">Choose a vector store to test search queries against</Text>
          </div>

          <Select
            value={selectedVectorStoreId}
            onChange={setSelectedVectorStoreId}
            placeholder="Select a vector store"
            size="large"
            style={{ width: "100%" }}
            showSearch
            optionFilterProp="children"
          >
            {vectorStores.map((vs) => (
              <Select.Option key={vs.vector_store_id} value={vs.vector_store_id}>
                <div className="flex flex-col">
                  <span className="font-medium">{vs.vector_store_name || vs.vector_store_id}</span>
                  {vs.vector_store_name && (
                    <span className="text-xs text-gray-500 font-mono">{vs.vector_store_id}</span>
                  )}
                </div>
              </Select.Option>
            ))}
          </Select>
        </div>
      </Card>

      {selectedVectorStoreId && (
        <VectorStoreTester vectorStoreId={selectedVectorStoreId} accessToken={accessToken} />
      )}
    </div>
  );
};

export default TestVectorStoreTab;
