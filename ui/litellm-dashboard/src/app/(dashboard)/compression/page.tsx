"use client";

import CompressionView from "./_components/CompressionView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CompressionPage() {
  const { accessToken } = useAuthorized();
  return <CompressionView accessToken={accessToken} />;
}
