import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Spin, Typography, Divider, Tooltip, Modal, message, Select, Form } from "antd";
import {
  PlusIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ClockIcon,
  CheckCircleIcon,
  SwitchHorizontalIcon,
} from "@heroicons/react/outline";
import { Policy, PolicyVersionListResponse } from "./types";
import VersionStatusBadge from "./version_status_badge";
import VersionComparison from "./version_comparison";
import {
  listPolicyVersions,
  updatePolicyVersionStatus,
  createPolicyVersion,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Text } = Typography;

interface VersionSidebarProps {
  policyName: string;
  currentPolicyId: string;
  accessToken: string | null;
  onVersionSelect: (version: Policy) => void;
  onVersionCreated: () => void;
}

const VersionSidebar: React.FC<VersionSidebarProps> = ({
  policyName,
  currentPolicyId,
  accessToken,
  onVersionSelect,
  onVersionCreated,
}) => {
  const [versions, setVersions] = useState<Policy[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [compareModalOpen, setCompareModalOpen] = useState(false);
  const [comparePolicyId1, setComparePolicyId1] = useState<string>("");
  const [comparePolicyId2, setComparePolicyId2] = useState<string>("");

  const loadVersions = async () => {
    if (!accessToken || !policyName) return;

    setIsLoading(true);
    try {
      const response: PolicyVersionListResponse = await listPolicyVersions(
        accessToken,
        policyName
      );
      setVersions(response.policies || []);
    } catch (error) {
      console.error("Failed to load versions:", error);
      NotificationsManager.fromBackend("Failed to load policy versions");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadVersions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [policyName, accessToken]);

  const handleCreateNewVersion = async () => {
    if (!accessToken) return;

    Modal.confirm({
      title: "Create New Version",
      content: `Create a new draft version from the current policy?`,
      okText: "Create",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading("create");
        try {
          await createPolicyVersion(accessToken, currentPolicyId);
          NotificationsManager.success("New version created successfully");
          await loadVersions();
          onVersionCreated();
        } catch (error) {
          console.error("Failed to create version:", error);
          NotificationsManager.fromBackend(
            "Failed to create version: " +
              (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handlePromoteToPublished = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;

    Modal.confirm({
      title: "Promote to Published",
      content: `Promote version ${versionNumber} to Published status? This makes it ready for testing.`,
      okText: "Promote",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "published");
          NotificationsManager.success("Version promoted to Published");
          await loadVersions();
        } catch (error) {
          console.error("Failed to promote version:", error);
          NotificationsManager.fromBackend(
            "Failed to promote version: " +
              (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handlePromoteToProduction = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;

    Modal.confirm({
      title: "Promote to Production",
      content: `Promote version ${versionNumber} to Production? This will make it the active version. Any existing production version will be demoted to Published.`,
      okText: "Promote",
      cancelText: "Cancel",
      okType: "primary",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "production");
          NotificationsManager.success("Version promoted to Production");
          await loadVersions();
        } catch (error) {
          console.error("Failed to promote version:", error);
          NotificationsManager.fromBackend(
            "Failed to promote version: " +
              (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const handleDemote = async (policyId: string, versionNumber: number) => {
    if (!accessToken) return;

    Modal.confirm({
      title: "Demote Version",
      content: `Demote version ${versionNumber} from Production to Published?`,
      okText: "Demote",
      cancelText: "Cancel",
      onOk: async () => {
        setActionLoading(policyId);
        try {
          await updatePolicyVersionStatus(accessToken, policyId, "published");
          NotificationsManager.success("Version demoted to Published");
          await loadVersions();
        } catch (error) {
          console.error("Failed to demote version:", error);
          NotificationsManager.fromBackend(
            "Failed to demote version: " +
              (error instanceof Error ? error.message : String(error))
          );
        } finally {
          setActionLoading(null);
        }
      },
    });
  };

  const formatDate = (dateString?: string | null) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const getActionButtons = (version: Policy) => {
    const isProcessing = actionLoading === version.policy_id;
    const status = version.version_status;

    if (status === "draft") {
      return (
        <Button
          size="xs"
          variant="secondary"
          icon={ChevronUpIcon}
          onClick={() => handlePromoteToPublished(version.policy_id, version.version_number || 1)}
          loading={isProcessing}
          disabled={!!actionLoading}
        >
          Publish
        </Button>
      );
    }

    if (status === "published") {
      return (
        <Button
          size="xs"
          variant="primary"
          icon={ChevronUpIcon}
          onClick={() =>
            handlePromoteToProduction(version.policy_id, version.version_number || 1)
          }
          loading={isProcessing}
          disabled={!!actionLoading}
        >
          To Production
        </Button>
      );
    }

    if (status === "production") {
      return (
        <Button
          size="xs"
          variant="secondary"
          icon={ChevronDownIcon}
          onClick={() => handleDemote(version.policy_id, version.version_number || 1)}
          loading={isProcessing}
          disabled={!!actionLoading}
        >
          Demote
        </Button>
      );
    }

    return null;
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-8">
        <Spin size="default" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with Create New Version button */}
      <div className="flex justify-between items-center">
        <Text strong style={{ fontSize: 16 }}>
          Versions
        </Text>
        <Button
          size="xs"
          icon={PlusIcon}
          onClick={handleCreateNewVersion}
          loading={actionLoading === "create"}
          disabled={!!actionLoading}
        >
          New Version
        </Button>
      </div>

      <Divider style={{ margin: "12px 0" }} />

      {/* Version List */}
      <div className="space-y-3">
        {versions.length === 0 ? (
          <Text type="secondary" style={{ fontSize: 13 }}>
            No versions found
          </Text>
        ) : (
          versions.map((version) => {
            const isActive = version.policy_id === currentPolicyId;
            const versionNumber = version.version_number || 1;
            const status = version.version_status || "draft";

            return (
              <div
                key={version.policy_id}
                className={`p-3 rounded-lg border transition-all cursor-pointer ${
                  isActive
                    ? "bg-blue-50 border-blue-300 shadow-sm"
                    : "bg-white border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => onVersionSelect(version)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Text strong style={{ fontSize: 14 }}>
                      v{versionNumber}
                    </Text>
                    {isActive && (
                      <CheckCircleIcon className="w-4 h-4 text-blue-500" />
                    )}
                  </div>
                  <VersionStatusBadge status={status as any} size="xs" />
                </div>

                <div className="flex items-center gap-1 mb-2">
                  <ClockIcon className="w-3 h-3 text-gray-400" />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {formatDate(version.created_at)}
                  </Text>
                </div>

                {version.description && (
                  <Text
                    type="secondary"
                    style={{
                      fontSize: 12,
                      display: "block",
                      marginBottom: 8,
                    }}
                    className="line-clamp-2"
                  >
                    {version.description}
                  </Text>
                )}

                <div className="flex justify-end">
                  {getActionButtons(version)}
                </div>
              </div>
            );
          })
        )}
      </div>

      <Divider style={{ margin: "16px 0" }} />

      {/* Compare versions */}
      {versions.length >= 2 && (
        <div className="mb-4">
          <Button
            size="xs"
            variant="secondary"
            icon={SwitchHorizontalIcon}
            onClick={() => {
              setComparePolicyId1(currentPolicyId);
              setComparePolicyId2(
                versions.find((v) => v.policy_id !== currentPolicyId)?.policy_id || ""
              );
              setCompareModalOpen(true);
            }}
          >
            Compare versions
          </Button>
        </div>
      )}

      {/* Silent Mirroring - Coming Soon */}
      <div
        className="p-4 rounded-lg"
        style={{
          backgroundColor: "#f9fafb",
          border: "1px dashed #d1d5db",
        }}
      >
        <div className="flex items-center justify-between mb-2">
          <Text strong style={{ fontSize: 14 }}>
            Silent Mirroring
          </Text>
          <span
            className="px-2 py-0.5 text-xs font-medium rounded"
            style={{
              backgroundColor: "#e0e7ff",
              color: "#4f46e5",
            }}
          >
            Coming Soon
          </span>
        </div>
        <Text type="secondary" style={{ fontSize: 12 }}>
          Test policy versions on production traffic without blocking requests.
          Shadow testing helps validate changes before full rollout.
        </Text>
      </div>

      {/* Compare versions modal */}
      <Modal
        title="Compare versions"
        open={compareModalOpen}
        onCancel={() => {
          setCompareModalOpen(false);
          setComparePolicyId1("");
          setComparePolicyId2("");
        }}
        footer={null}
        width={720}
        destroyOnHidden
      >
        <Form layout="vertical" className="mb-4">
          <div className="grid grid-cols-2 gap-4">
            <Form.Item label="Version A">
              <Select
                placeholder="Select version"
                value={comparePolicyId1 || undefined}
                onChange={setComparePolicyId1}
                style={{ width: "100%" }}
                options={versions.map((v) => ({
                  label: `v${v.version_number || 1} (${v.version_status || "draft"})`,
                  value: v.policy_id,
                }))}
              />
            </Form.Item>
            <Form.Item label="Version B">
              <Select
                placeholder="Select version"
                value={comparePolicyId2 || undefined}
                onChange={setComparePolicyId2}
                style={{ width: "100%" }}
                options={versions.map((v) => ({
                  label: `v${v.version_number || 1} (${v.version_status || "draft"})`,
                  value: v.policy_id,
                }))}
              />
            </Form.Item>
          </div>
        </Form>
        {comparePolicyId1 && comparePolicyId2 && comparePolicyId1 !== comparePolicyId2 && (
          <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
            <VersionComparison
              policyId1={comparePolicyId1}
              policyId2={comparePolicyId2}
              accessToken={accessToken}
            />
          </div>
        )}
        {comparePolicyId1 && comparePolicyId2 && comparePolicyId1 === comparePolicyId2 && (
          <Text type="secondary">Select two different versions to compare.</Text>
        )}
      </Modal>
    </div>
  );
};

export default VersionSidebar;
