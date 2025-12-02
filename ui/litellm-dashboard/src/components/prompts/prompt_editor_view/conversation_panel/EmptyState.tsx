import React from "react";
import { RobotOutlined } from "@ant-design/icons";

interface EmptyStateProps {
  hasVariables: boolean;
}

const EmptyState: React.FC<EmptyStateProps> = ({ hasVariables }) => {
  return (
    <div className="h-full flex flex-col items-center justify-center text-gray-400">
      <RobotOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
      <span className="text-base">
        {hasVariables
          ? "Fill in the variables above, then type a message to start testing"
          : "Type a message below to start testing your prompt"}
      </span>
    </div>
  );
};

export default EmptyState;

