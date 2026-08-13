import { useCloudZeroSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Card, CardContent } from "@/components/ui/card";
import CloudZeroEmptyPlaceholder from "./CloudZeroEmptyPlaceholder";
import { useState } from "react";
import CloudZeroCreationModal from "./CloudZeroCreateModal";
import { useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { CloudZeroIntegrationSettings } from "./CloudZeroIntegrationSettings";

export default function CloudZeroCostTracking() {
  const { accessToken } = useAuthorized();
  const { data: settings, isLoading, error } = useCloudZeroSettings(accessToken);
  const queryClient = useQueryClient();
  const cloudZeroSettingsKeys = createQueryKeys("cloudZeroSettings");

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const handleCreateModalOk = async () => {
    setIsCreateModalOpen(false);
    await queryClient.invalidateQueries({ queryKey: cloudZeroSettingsKeys.list({}) });
  };

  const handleCreateModalCancel = () => {
    setIsCreateModalOpen(false);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading CloudZero settings...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-destructive">
            Error loading CloudZero settings: {error instanceof Error ? error.message : String(error)}
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!settings) {
    return (
      <>
        <CloudZeroEmptyPlaceholder startCreation={() => setIsCreateModalOpen(true)} />
        <CloudZeroCreationModal
          open={isCreateModalOpen}
          onOk={handleCreateModalOk}
          onCancel={handleCreateModalCancel}
        />
      </>
    );
  }

  return (
    <>
      <CloudZeroIntegrationSettings settings={settings} onSettingsUpdated={handleCreateModalOk} />
    </>
  );
}
