"use client";

import React from "react";
import { Form } from "antd";

import AddAutoRouterTab from "@/components/add_model/add_auto_router_tab";

interface AutorouterTabProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

const AutorouterTab: React.FC<AutorouterTabProps> = ({ accessToken, userRole }) => {
  const [form] = Form.useForm();

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full">
      <AddAutoRouterTab form={form} handleOk={() => form.resetFields()} accessToken={accessToken} userRole={userRole} />
    </div>
  );
};

export default AutorouterTab;
