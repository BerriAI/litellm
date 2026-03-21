import { useCloudZeroSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Card, Typography } from "antd";
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
        <Typography.Text>Loading CloudZero settings...</Typography.Text>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Typography.Text className="text-red-600">
          Error loading CloudZero settings: {error instanceof Error ? error.message : String(error)}
        </Typography.Text>
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
