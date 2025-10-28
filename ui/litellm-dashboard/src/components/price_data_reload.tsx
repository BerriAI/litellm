import React, { useState, useEffect } from "react";
import { Button, Popconfirm, Modal, InputNumber, Space, Typography, Tag, Card } from "antd";
import { ReloadOutlined, ClockCircleOutlined, StopOutlined } from "@ant-design/icons";
import {
  reloadModelCostMap,
  scheduleModelCostMapReload,
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

const { Text } = Typography;

interface ReloadStatus {
  scheduled: boolean;
  interval_hours: number | null;
  last_run: string | null;
  next_run: string | null;
}

interface PriceDataReloadProps {
  accessToken: string;
  onReloadSuccess?: () => void;
  buttonText?: string;
  showIcon?: boolean;
  size?: "small" | "middle" | "large";
  type?: "primary" | "default" | "dashed" | "link" | "text";
  className?: string;
}

const PriceDataReload: React.FC<PriceDataReloadProps> = ({
  accessToken,
  onReloadSuccess,
  buttonText = "Reload Price Data",
  showIcon = true,
  size = "middle",
  type = "primary",
  className = "",
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isScheduling, setIsScheduling] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [hours, setHours] = useState<number>(6);
  const [reloadStatus, setReloadStatus] = useState<ReloadStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(false);

  // Fetch status on component mount and periodically
  useEffect(() => {
    fetchReloadStatus();

    // Refresh status every 30 seconds to keep it up to date
    const interval = setInterval(() => {
      fetchReloadStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, [accessToken]);

  const fetchReloadStatus = async () => {
    if (!accessToken) return;

    setLoadingStatus(true);
    try {
      console.log("Fetching reload status...");
      const status = await getModelCostMapReloadStatus(accessToken);
      console.log("Received status:", status);
      setReloadStatus(status);
    } catch (error) {
      console.error("Failed to fetch reload status:", error);
      // Set a default status to prevent UI issues
      setReloadStatus({
        scheduled: false,
        interval_hours: null,
        last_run: null,
        next_run: null,
      });
    } finally {
      setLoadingStatus(false);
    }
  };

  const handleHardRefresh = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);

      if (response.status === "success") {
        NotificationsManager.success(`Price data reloaded successfully! ${response.models_count || 0} models updated.`);
        onReloadSuccess?.();
        // Refresh status after successful reload
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("Failed to reload price data");
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      NotificationsManager.fromBackend("Failed to reload price data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  const handleScheduleReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    if (hours <= 0) {
      NotificationsManager.fromBackend("Hours must be greater than 0");
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, hours);

      if (response.status === "success") {
        NotificationsManager.success(`Periodic reload scheduled for every ${hours} hours`);
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("Failed to schedule periodic reload");
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      NotificationsManager.fromBackend("Failed to schedule periodic reload. Please try again.");
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);

      if (response.status === "success") {
        NotificationsManager.success("Periodic reload cancelled successfully");
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("Failed to cancel periodic reload");
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      NotificationsManager.fromBackend("Failed to cancel periodic reload. Please try again.");
    } finally {
      setIsCancelling(false);
    }
  };

  const formatDateTime = (dateTimeString: string | null) => {
    if (!dateTimeString) return "Never";
    try {
      return new Date(dateTimeString).toLocaleString();
    } catch {
      return dateTimeString;
    }
  };

  const getStatusText = () => {
    if (!reloadStatus?.scheduled) return "Not scheduled";
    if (!reloadStatus.last_run) return "Ready";
    return "Active";
  };

  const getStatusColor = () => {
    if (!reloadStatus?.scheduled) return "default";
    if (!reloadStatus.last_run) return "processing";
    return "success";
  };

  return (
    <div className={className}>
      {/* Action Buttons */}
      <Space direction="horizontal" size="middle" style={{ marginBottom: 16 }}>
        {/* Hard Refresh Button - Always visible */}
        <Popconfirm
          title="Hard Refresh Price Data"
          description="This will immediately fetch the latest pricing information from the remote source. Continue?"
          onConfirm={handleHardRefresh}
          okText="Yes"
          cancelText="No"
          okButtonProps={{
            style: {
              backgroundColor: "#6366f1",
              borderColor: "#6366f1",
              color: "white",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
              transition: "all 0.2s ease-in-out",
            },
            onMouseEnter: (e) => {
              e.currentTarget.style.backgroundColor = "#4f46e5";
            },
            onMouseLeave: (e) => {
              e.currentTarget.style.backgroundColor = "#6366f1";
            },
          }}
        >
          <Button
            type={type}
            size={size}
            loading={isLoading}
            icon={showIcon ? <ReloadOutlined /> : undefined}
            style={{
              backgroundColor: "#6366f1",
              borderColor: "#6366f1",
              color: "white",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
              transition: "all 0.2s ease-in-out",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "#4f46e5";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "#6366f1";
            }}
          >
            {buttonText}
          </Button>
        </Popconfirm>

        {/* Periodic Reload Controls */}
        {!reloadStatus?.scheduled ? (
          <Button
            type="default"
            size={size}
            icon={<ClockCircleOutlined />}
            onClick={() => setShowScheduleModal(true)}
            style={{
              borderColor: "#d9d9d9",
              color: "#6366f1",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
            }}
          >
            Set Up Periodic Reload
          </Button>
        ) : (
          <Button
            type="default"
            size={size}
            danger
            icon={<StopOutlined />}
            loading={isCancelling}
            onClick={handleCancelReload}
            style={{
              borderColor: "#ff4d4f",
              color: "#ff4d4f",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
            }}
          >
            Cancel Periodic Reload
          </Button>
        )}
      </Space>

      {/* Status Card */}
      {reloadStatus && (
        <Card
          size="small"
          style={{
            backgroundColor: "#f8f9fa",
            border: "1px solid #e9ecef",
            borderRadius: 8,
          }}
        >
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            {reloadStatus.scheduled ? (
              <div>
                <Tag color="green" icon={<ClockCircleOutlined />}>
                  Scheduled every {reloadStatus.interval_hours} hours
                </Tag>
              </div>
            ) : (
              <Text type="secondary">No periodic reload scheduled</Text>
            )}

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                Last run:
              </Text>
              <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.last_run)}</Text>
            </div>

            {reloadStatus.scheduled && (
              <>
                {reloadStatus.next_run && (
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text type="secondary" style={{ fontSize: "12px" }}>
                      Next run:
                    </Text>
                    <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.next_run)}</Text>
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    Status:
                  </Text>
                  <Tag color={getStatusColor()}>{getStatusText()}</Tag>
                </div>
              </>
            )}
          </Space>
        </Card>
      )}

      {/* Schedule Modal */}
      <Modal
        title="Set Up Periodic Reload"
        open={showScheduleModal}
        onOk={handleScheduleReload}
        onCancel={() => setShowScheduleModal(false)}
        confirmLoading={isScheduling}
        okText="Schedule"
        cancelText="Cancel"
        okButtonProps={{
          style: {
            backgroundColor: "#6366f1",
            borderColor: "#6366f1",
            color: "white",
          },
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <Text>Set up automatic reload of price data every:</Text>
        </div>
        <div style={{ marginBottom: 16 }}>
          <InputNumber
            min={1}
            max={168} // 1 week max
            value={hours}
            onChange={(value) => setHours(value || 6)}
            addonAfter="hours"
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <Text type="secondary">
            This will automatically fetch the latest pricing data from the remote source every {hours} hours.
          </Text>
        </div>
      </Modal>
    </div>
  );
};

export default PriceDataReload;
