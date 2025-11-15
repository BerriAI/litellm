"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useListKeys } from "@/hooks/keys/useKeys";
import { Card, Title, Text } from "@tremor/react";

const KeysPage = () => {
  const { accessToken, userRole, userId, userEmail } = useAuthorized();
  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <KeysPageContent accessToken={accessToken} userRole={userRole} userId={userId} userEmail={userEmail} />
    </QueryClientProvider>
  );
};

interface KeysPageContentProps {
  accessToken: string | null;
  userRole: string | null;
  userId: string | null;
  userEmail: string | null;
}

const KeysPageContent = ({ accessToken, userRole, userId, userEmail }: KeysPageContentProps) => {
  const { keys, isLoading, error, pagination } = useListKeys({
    accessToken: accessToken || "",
    page: 1,
    pageSize: 100,
  });

  return (
    <div className="w-full mx-auto px-8 py-8">
      <div className="mb-6">
        <Title>Virtual Keys (New)</Title>
        <Text className="mt-2">
          This is the new Virtual Keys page using React Query. The page structure supports nested routes like
          /ui/keys/&#123;key_uuid&#125; for individual key details.
        </Text>
      </div>

      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Title>Keys Overview</Title>
            <Text>{pagination.totalCount} total keys</Text>
          </div>

          {isLoading && (
            <div className="text-center py-8">
              <Text>Loading keys...</Text>
            </div>
          )}

          {error && (
            <div className="text-center py-8">
              <Text className="text-red-500">Error loading keys: {error.message}</Text>
            </div>
          )}

          {!isLoading && !error && keys.length === 0 && (
            <div className="text-center py-8">
              <Text>No keys found</Text>
            </div>
          )}

          {!isLoading && !error && keys.length > 0 && (
            <div className="space-y-2">
              <div className="grid grid-cols-4 gap-4 font-semibold pb-2 border-b">
                <div>Key Name</div>
                <div>Token</div>
                <div>Spend</div>
                <div>Created</div>
              </div>
              {keys.slice(0, 10).map((key) => (
                <div key={key.token_id} className="grid grid-cols-4 gap-4 py-2 border-b hover:bg-gray-50">
                  <div className="truncate">{key.key_name || "Unnamed"}</div>
                  <div className="truncate font-mono text-sm">{key.token.substring(0, 20)}...</div>
                  <div>${key.spend.toFixed(4)}</div>
                  <div>{new Date(key.created_at).toLocaleDateString()}</div>
                </div>
              ))}
              {keys.length > 10 && (
                <div className="pt-4 text-center">
                  <Text>Showing 10 of {keys.length} keys</Text>
                </div>
              )}
            </div>
          )}

          <div className="pt-4 border-t mt-4">
            <Text className="text-sm text-gray-500">
              User: {userEmail || userId} | Role: {userRole}
            </Text>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default KeysPage;
