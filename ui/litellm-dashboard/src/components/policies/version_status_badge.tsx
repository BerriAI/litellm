import React from "react";
import { Badge } from "@tremor/react";

export type VersionStatus = "draft" | "published" | "production";

interface VersionStatusBadgeProps {
  status: VersionStatus;
  size?: "xs" | "sm" | "md" | "lg";
}

const VersionStatusBadge: React.FC<VersionStatusBadgeProps> = ({
  status,
  size = "sm",
}) => {
  const getStatusConfig = (status: VersionStatus) => {
    switch (status) {
      case "draft":
        return {
          color: "blue" as const,
          label: "Draft",
        };
      case "published":
        return {
          color: "yellow" as const,
          label: "Published",
        };
      case "production":
        return {
          color: "green" as const,
          label: "Production",
        };
      default:
        return {
          color: "gray" as const,
          label: "Unknown",
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <Badge color={config.color} size={size}>
      {config.label}
    </Badge>
  );
};

export default VersionStatusBadge;
