"use client";

import React from "react";
import { Title, Text } from "@tremor/react";
import PolicyComplianceTab from "@/components/UsagePage/components/PolicyComplianceTab";

const CompliancePage = () => {
  return (
    <div style={{ width: "100%" }} className="p-8">
      <Title>Policy & Compliance View</Title>
      <Text className="mb-6">Monitor regulatory compliance across all AI requests</Text>
      <PolicyComplianceTab />
    </div>
  );
};

export default CompliancePage;
