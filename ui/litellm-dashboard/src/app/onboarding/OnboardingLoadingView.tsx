import React from "react";
import { Spin } from "antd";
import { LoadingOutlined } from "@ant-design/icons";

export function OnboardingLoadingView() {
  return (
    <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
      <Spin indicator={<LoadingOutlined spin />} size="large" />
    </div>
  );
}
