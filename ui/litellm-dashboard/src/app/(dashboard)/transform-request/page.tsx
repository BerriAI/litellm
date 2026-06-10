"use client";

import TransformRequestPanel from "@/components/transform_request";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TransformRequestRoute() {
  const { accessToken } = useAuthorized();
  return <TransformRequestPanel accessToken={accessToken} />;
}
