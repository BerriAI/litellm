import React, { useState } from "react";
import { Button, message, Popconfirm, Tooltip } from "antd";
import { reloadModelCostMap } from "./networking";

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

  const handleReload = async () => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);
      
      if (response.status === "success") {
        message.success(
          `Price data reloaded successfully! ${response.models_count || 0} models updated.`
        );
        onReloadSuccess?.();
      } else {
        message.error("Failed to reload price data");
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      message.error("Failed to reload price data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Popconfirm
      title="Reload Price Data"
      description="This will fetch the latest pricing information from the remote source. Continue?"
      onConfirm={handleReload}
      okText="Yes"
      cancelText="No"
    >
      <Tooltip title="Reload latest pricing data from remote source">
        <Button
          type={type}
          size={size}
          loading={isLoading}
          className={className}
          style={{
            backgroundColor: "#6366f1", // Tremor primary color
            borderColor: "#6366f1",
            color: "white",
            fontWeight: "500",
            borderRadius: "0.5rem",
            padding: "0.5rem 1rem",
            height: "auto",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
            fontSize: "0.875rem",
            lineHeight: "1.25rem",
            transition: "all 0.2s ease-in-out",
            boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = "#4f46e5";
            e.currentTarget.style.borderColor = "#4f46e5";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "#6366f1";
            e.currentTarget.style.borderColor = "#6366f1";
          }}
        >
          {showIcon ? "↻ " : ""}{buttonText}
        </Button>
      </Tooltip>
    </Popconfirm>
  );
};

export default PriceDataReload; 