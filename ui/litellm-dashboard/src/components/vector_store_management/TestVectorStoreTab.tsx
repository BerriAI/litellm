import React, { useState } from "react";
import { Select } from "antd";
import { Card } from "@/components/ui/card";
import { VectorStoreTester } from "./VectorStoreTester";
import { VectorStore } from "./types";

interface TestVectorStoreTabProps {
  accessToken: string | null;
  vectorStores: VectorStore[];
}

const TestVectorStoreTab: React.FC<TestVectorStoreTabProps> = ({
  accessToken,
  vectorStores,
}) => {
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<
    string | undefined
  >(vectorStores.length > 0 ? vectorStores[0].vector_store_id : undefined);

  if (!accessToken) {
    return (
      <Card className="p-4">
        <p className="text-muted-foreground">
          Access token is required to test vector stores.
        </p>
      </Card>
    );
  }

  if (vectorStores.length === 0) {
    return (
      <Card className="p-4">
        <div className="text-center py-8">
          <p className="text-muted-foreground">
            No vector stores available. Create one first to test it.
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <div className="space-y-4">
          <div>
            <h5 className="text-base font-semibold">Select Vector Store</h5>
            <p className="text-muted-foreground text-sm">
              Choose a vector store to test search queries against
            </p>
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
              <Select.Option
                key={vs.vector_store_id}
                value={vs.vector_store_id}
              >
                <div className="flex flex-col">
                  <span className="font-medium">
                    {vs.vector_store_name || vs.vector_store_id}
                  </span>
                  {vs.vector_store_name && (
                    <span className="text-xs text-muted-foreground font-mono">
                      {vs.vector_store_id}
                    </span>
                  )}
                </div>
              </Select.Option>
            ))}
          </Select>
        </div>
      </Card>

      {selectedVectorStoreId && (
        <VectorStoreTester
          vectorStoreId={selectedVectorStoreId}
          accessToken={accessToken}
        />
      )}
    </div>
  );
};

export default TestVectorStoreTab;
