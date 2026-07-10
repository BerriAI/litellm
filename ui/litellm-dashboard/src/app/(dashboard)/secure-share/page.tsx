"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { isAdminRole } from "@/utils/roles";
import { Card, Typography } from "antd";
import { Suspense } from "react";
import SecureShareCreate from "./_components/SecureShareCreate";

const { Title, Paragraph } = Typography;

function SecureSharePageContent() {
  const { isLoading, isAuthorized, accessToken, userRole } = useAuthorized();
  if (isLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  if (!userRole || !isAdminRole(userRole)) {
    return (
      <Card>
        <Title level={4}>Secure Share</Title>
        <Paragraph>Only proxy admins can create secure shares.</Paragraph>
      </Card>
    );
  }
  return <SecureShareCreate accessToken={accessToken} />;
}

export default function SecureSharePage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <SecureSharePageContent />
    </Suspense>
  );
}
