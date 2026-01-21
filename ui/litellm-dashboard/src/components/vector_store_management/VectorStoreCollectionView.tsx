import React, { useEffect, useState } from "react";
import {
  Button as TremorButton,
  Card,
  Title,
  Text,
  Table,
  TableHead,
  TableHeaderCell,
  TableRow,
  TableBody,
  TableCell,
} from "@tremor/react";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { qdrantCollectionInfoCall, qdrantCollectionPointsCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreCollectionViewProps {
  vectorStoreId: string;
  accessToken: string | null;
  onClose: () => void;
}

const VectorStoreCollectionView: React.FC<VectorStoreCollectionViewProps> = ({
  vectorStoreId,
  accessToken,
  onClose,
}) => {
  const [collectionInfo, setCollectionInfo] = useState<Record<string, any> | null>(null);
  const [points, setPoints] = useState<Array<Record<string, any>>>([]);
  const [nextOffset, setNextOffset] = useState<Record<string, any> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const fetchCollectionInfo = async () => {
    if (!accessToken) return;
    try {
      const response = await qdrantCollectionInfoCall(accessToken, vectorStoreId);
      setCollectionInfo(response);
    } catch (error) {
      console.error("Error fetching collection info:", error);
      NotificationsManager.fromBackend("Error fetching collection info: " + error);
    }
  };

  const fetchCollectionPoints = async (offset?: Record<string, any>) => {
    if (!accessToken) return;
    try {
      const response = await qdrantCollectionPointsCall(accessToken, {
        vector_store_id: vectorStoreId,
        limit: 20,
        offset,
      });
      const newPoints = response?.result?.points || [];
      const newOffset = response?.result?.next_page_offset || null;
      setPoints((current) => (offset ? [...current, ...newPoints] : newPoints));
      setNextOffset(newOffset);
    } catch (error) {
      console.error("Error fetching collection points:", error);
      NotificationsManager.fromBackend("Error fetching collection points: " + error);
    }
  };

  useEffect(() => {
    const load = async () => {
      setIsLoading(true);
      await Promise.all([fetchCollectionInfo(), fetchCollectionPoints()]);
      setIsLoading(false);
    };
    load();
  }, [accessToken, vectorStoreId]);

  const handleLoadMore = async () => {
    if (!nextOffset) return;
    setIsLoadingMore(true);
    await fetchCollectionPoints(nextOffset);
    setIsLoadingMore(false);
  };

  return (
    <div className="w-full mx-4 h-[75vh]">
      <div className="gap-2 p-8 h-[75vh] w-full mt-2">
        <div className="flex items-center justify-between mb-4">
          <div>
            <Title>Collection View</Title>
            <Text className="text-gray-500">Vector Store ID: {vectorStoreId}</Text>
          </div>
          <TremorButton
            variant="secondary"
            icon={ArrowLeftIcon}
            onClick={onClose}
            className="h-8"
          >
            Back to Vector Stores
          </TremorButton>
        </div>

        <Card className="mb-4">
          <Title className="mb-2">Collection Info</Title>
          {isLoading ? (
            <Text>Loading collection info...</Text>
          ) : collectionInfo ? (
            <pre className="text-xs whitespace-pre-wrap text-gray-600">
              {JSON.stringify(collectionInfo?.result || collectionInfo, null, 2)}
            </pre>
          ) : (
            <Text className="text-gray-500">No collection info available.</Text>
          )}
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-2">
            <Title>Collection Points</Title>
            <Text className="text-gray-500 text-xs">{points.length} points loaded</Text>
          </div>
          {isLoading && points.length === 0 ? (
            <Text>Loading points...</Text>
          ) : points.length === 0 ? (
            <Text className="text-gray-500">No points found.</Text>
          ) : (
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-1 [&_th]:py-1">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>ID</TableHeaderCell>
                    <TableHeaderCell>Payload</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {points.map((point) => (
                    <TableRow key={String(point.id)}>
                      <TableCell className="font-mono text-xs">{String(point.id)}</TableCell>
                      <TableCell>
                        <pre className="text-xs whitespace-pre-wrap text-gray-600">
                          {JSON.stringify(point.payload || {}, null, 2)}
                        </pre>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
          {nextOffset && (
            <div className="mt-3">
              <TremorButton
                variant="secondary"
                onClick={handleLoadMore}
                loading={isLoadingMore}
                disabled={isLoadingMore}
              >
                Load more
              </TremorButton>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default VectorStoreCollectionView;
