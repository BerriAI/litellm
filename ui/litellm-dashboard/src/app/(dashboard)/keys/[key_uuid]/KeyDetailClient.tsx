"use client";

import { useParams, useRouter } from "next/navigation";
import { Card, Title, Text, Button } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const KeyDetailClient = () => {
  const params = useParams();
  const router = useRouter();
  const { accessToken, userRole } = useAuthorized();
  const keyUuid = params.key_uuid as string;

  const handleBackToKeys = () => {
    router.push("/keys");
  };

  return (
    <div className="w-full mx-auto px-8 py-8">
      <div className="mb-6">
        <Button size="xs" variant="light" onClick={handleBackToKeys}>
          ← Back to Keys
        </Button>
      </div>

      <div className="mb-6">
        <Title>Key Details</Title>
        <Text className="mt-2">Viewing details for key: {keyUuid}</Text>
      </div>

      <Card>
        <div className="space-y-4">
          <div>
            <Text className="font-semibold">Key UUID</Text>
            <Text className="font-mono mt-1">{keyUuid}</Text>
          </div>

          <div className="border-t pt-4">
            <Text className="font-semibold">User Info</Text>
            <div className="mt-2 space-y-1">
              <Text className="text-sm">Role: {userRole}</Text>
              <Text className="text-sm">Has Access Token: {accessToken ? "Yes" : "No"}</Text>
            </div>
          </div>

          <div className="border-t pt-4">
            <Title className="text-lg">Coming Soon</Title>
            <div className="mt-4 space-y-2">
              <Text>• Key metadata and configuration</Text>
              <Text>• Usage statistics and spend breakdown</Text>
              <Text>• Rate limits and quotas</Text>
              <Text>• Associated team and organization</Text>
              <Text>• Audit log and activity history</Text>
              <Text>• Edit key settings</Text>
              <Text>• Rotate or revoke key</Text>
            </div>
          </div>
        </div>
      </Card>

      <Card className="mt-6">
        <Title>Implementation Notes</Title>
        <div className="mt-4 space-y-2">
          <Text>• Create a useKey hook to fetch individual key details</Text>
          <Text>• Add mutations for key updates (useMutation)</Text>
          <Text>• Integrate with existing key management components</Text>
          <Text>• Add proper error handling and loading states</Text>
        </div>
      </Card>
    </div>
  );
};

export default KeyDetailClient;
