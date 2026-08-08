"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ActiveRequests from "@/components/ActiveRequests/ActiveRequests";

export default function ActiveRequestsPage() {
  const { accessToken } = useAuthorized();
  return <ActiveRequests accessToken={accessToken} />;
}
