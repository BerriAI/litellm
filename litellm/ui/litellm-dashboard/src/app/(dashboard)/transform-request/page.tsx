"use client";

import TransformRequestPanel from "./TransformRequestPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TransformRequest() {
  const { accessToken } = useAuthorized();
  return <TransformRequestPanel accessToken={accessToken} />;
}
