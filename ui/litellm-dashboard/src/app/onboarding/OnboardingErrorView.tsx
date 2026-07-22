import React from "react";
import { Alert, Button } from "antd";
import { getLoginUrl } from "@/utils/returnUrlUtils";

export function OnboardingErrorView() {
  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Alert
        type="error"
        message="Failed to load invitation"
        description="The invitation link may be invalid or expired."
        showIcon
      />
      <div className="mt-4">
        <Button href={getLoginUrl()}>Back to Login</Button>
      </div>
    </div>
  );
}
