import React, { useState, useEffect } from "react";
import { Button, Popconfirm, Modal, InputNumber, Space, Typography, Tag, Card, Tooltip, Divider } from "antd";
import {
  ReloadOutlined,
  ClockCircleOutlined,
  StopOutlined,
  CloudOutlined,
  DatabaseOutlined,
  InfoCircleOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import {
  reloadModelCostMap,
  scheduleModelCostMapReload,
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
  getModelCostMapSource,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

interface ReloadStatus {
  scheduled: boolean;
  interval_hours: number | null;
  last_run: string | null;
  next_run: string | null;
}

interface CostMapSourceInfo {
  source: "local" | "remote";
  url: string | null;
  is_env_forced: boolean;
  fallback_reason: string | null;
  model_count: number;
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
  buttonText,
  showIcon = true,
  size = "middle",
  type = "primary",
  className = "",
}) => {
  const { t } = useTranslation("gateway");
  const [isLoading, setIsLoading] = useState(false);
  const [isScheduling, setIsScheduling] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [hours, setHours] = useState<number>(6);
  const [reloadStatus, setReloadStatus] = useState<ReloadStatus | null>(null);
  const [sourceInfo, setSourceInfo] = useState<CostMapSourceInfo | null>(null);

  // Fetch status on component mount and periodically
  useEffect(() => {
    fetchReloadStatus();
    fetchSourceInfo();

    // Refresh status every 30 seconds to keep it up to date
    const interval = setInterval(() => {
      fetchReloadStatus();
      fetchSourceInfo();
    }, 30000);

    return () => clearInterval(interval);
  }, [accessToken]);

  const fetchReloadStatus = async () => {
    if (!accessToken) return;

    try {
      const status = await getModelCostMapReloadStatus(accessToken);
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
    }
  };

  const fetchSourceInfo = async () => {
    if (!accessToken) return;

    try {
      const info = await getModelCostMapSource(accessToken);
      setSourceInfo(info);
    } catch (error) {
      console.error("Failed to fetch cost map source info:", error);
    }
  };

  const handleHardRefresh = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("models.priceData.noToken"));
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);

      if (response.status === "success") {
        NotificationsManager.success(t("models.priceData.reloaded", { count: response.models_count || 0 }));
        onReloadSuccess?.();
        // Refresh status and source info after successful reload
        await fetchReloadStatus();
        await fetchSourceInfo();
      } else {
        NotificationsManager.fromBackend(t("models.priceData.reloadFailed"));
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      NotificationsManager.fromBackend(t("models.priceData.reloadRetry"));
    } finally {
      setIsLoading(false);
    }
  };
  const handleScheduleReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("models.priceData.noToken"));
      return;
    }

    if (hours <= 0) {
      NotificationsManager.fromBackend(t("models.priceData.invalidHours"));
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, hours);

      if (response.status === "success") {
        NotificationsManager.success(t("models.priceData.scheduled", { hours }));
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend(t("models.priceData.scheduleFailed"));
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      NotificationsManager.fromBackend(t("models.priceData.scheduleRetry"));
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("models.priceData.noToken"));
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);

      if (response.status === "success") {
        NotificationsManager.success(t("models.priceData.cancelled"));
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend(t("models.priceData.cancelFailed"));
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      NotificationsManager.fromBackend(t("models.priceData.cancelRetry"));
    } finally {
      setIsCancelling(false);
    }
  };

  const formatDateTime = (dateTimeString: string | null) => {
    if (!dateTimeString) return t("models.priceData.never");
    try {
      return new Date(dateTimeString).toLocaleString();
    } catch {
      return dateTimeString;
    }
  };

  const getStatusText = () => {
    if (!reloadStatus?.scheduled) return t("models.priceData.statusNotScheduled");
    if (!reloadStatus.last_run) return t("models.priceData.statusReady");
    return t("models.priceData.statusActive");
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
          title={t("models.priceData.hardRefreshTitle")}
          description={t("models.priceData.hardRefreshDescription")}
          onConfirm={handleHardRefresh}
          okText={t("models.priceData.yes")}
          cancelText={t("models.priceData.no")}
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
            {buttonText ?? t("models.priceData.reload")}
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
            {t("models.priceData.setup")}
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
            {t("models.priceData.cancelPeriodic")}
          </Button>
        )}
      </Space>

      {/* Cost Map Source Info Card */}
      {sourceInfo && (
        <Card
          size="small"
          style={{
            backgroundColor: sourceInfo.source === "remote" ? "#f0f7ff" : "#fff8f0",
            border: `1px solid ${sourceInfo.source === "remote" ? "#bae0ff" : "#ffd591"}`,
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            {/* Header row */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {sourceInfo.source === "remote" ? (
                <CloudOutlined style={{ color: "#1677ff", fontSize: 16 }} />
              ) : (
                <DatabaseOutlined style={{ color: "#fa8c16", fontSize: 16 }} />
              )}
              <Text strong style={{ fontSize: "13px" }}>
                {t("models.priceData.source")}
              </Text>
              <Tag
                color={sourceInfo.source === "remote" ? "blue" : "orange"}
                style={{ marginLeft: "auto", fontWeight: 600, textTransform: "uppercase", fontSize: "11px" }}
              >
                {t(sourceInfo.source === "remote" ? "models.priceData.remote" : "models.priceData.local")}
              </Tag>
            </div>

            <Divider style={{ margin: "6px 0" }} />

            {/* Model count */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                {t("models.priceData.modelsLoaded")}
              </Text>
              <Text strong style={{ fontSize: "12px" }}>
                {sourceInfo.model_count.toLocaleString()}
              </Text>
            </div>

            {/* URL (when remote or attempted) */}
            {sourceInfo.url && (
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                <Text type="secondary" style={{ fontSize: "12px", whiteSpace: "nowrap" }}>
                  {t(sourceInfo.source === "remote" ? "models.priceData.loadedFrom" : "models.priceData.attemptedUrl")}
                </Text>
                <Tooltip title={sourceInfo.url}>
                  <Text
                    style={{
                      fontSize: "11px",
                      maxWidth: 240,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      display: "block",
                      color: "#1677ff",
                      cursor: "default",
                    }}
                  >
                    {sourceInfo.url}
                  </Text>
                </Tooltip>
              </div>
            )}

            {/* Env forced notice */}
            {sourceInfo.is_env_forced && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
                <InfoCircleOutlined style={{ color: "#fa8c16", fontSize: 12 }} />
                <Text type="secondary" style={{ fontSize: "11px" }}>
                  {t("models.priceData.localForced")} <code>LITELLM_LOCAL_MODEL_COST_MAP=True</code>
                </Text>
              </div>
            )}

            {/* Fallback reason */}
            {sourceInfo.fallback_reason && (
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 6,
                  backgroundColor: "#fff7e6",
                  border: "1px solid #ffd591",
                  borderRadius: 4,
                  padding: "4px 8px",
                  marginTop: 2,
                }}
              >
                <WarningOutlined style={{ color: "#fa8c16", fontSize: 12, marginTop: 2 }} />
                <Text style={{ fontSize: "11px", color: "#614700" }}>
                  {t("models.priceData.fallback", { reason: sourceInfo.fallback_reason })}
                </Text>
              </div>
            )}
          </Space>
        </Card>
      )}

      {/* Reload Schedule Status Card */}
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
                  {t("models.priceData.scheduledEvery", { hours: reloadStatus.interval_hours })}
                </Tag>
              </div>
            ) : (
              <Text type="secondary">{t("models.priceData.notScheduled")}</Text>
            )}

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                {t("models.priceData.lastRun")}
              </Text>
              <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.last_run)}</Text>
            </div>

            {reloadStatus.scheduled && (
              <>
                {reloadStatus.next_run && (
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text type="secondary" style={{ fontSize: "12px" }}>
                      {t("models.priceData.nextRun")}
                    </Text>
                    <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.next_run)}</Text>
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    {t("models.priceData.status")}
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
        title={t("models.priceData.setup")}
        open={showScheduleModal}
        onOk={handleScheduleReload}
        onCancel={() => setShowScheduleModal(false)}
        confirmLoading={isScheduling}
        okText={t("models.priceData.schedule")}
        cancelText={t("models.priceData.cancel")}
        okButtonProps={{
          style: {
            backgroundColor: "#6366f1",
            borderColor: "#6366f1",
            color: "white",
          },
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <Text>{t("models.priceData.setupDescription")}</Text>
        </div>
        <div style={{ marginBottom: 16 }}>
          <InputNumber
            min={1}
            max={168} // 1 week max
            value={hours}
            onChange={(value) => setHours(value || 6)}
            addonAfter={t("models.priceData.hours")}
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <Text type="secondary">{t("models.priceData.scheduleDescription", { hours })}</Text>
        </div>
      </Modal>
    </div>
  );
};

export default PriceDataReload;
