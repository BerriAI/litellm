"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { EmbeddedLens } from "@/components/lens/EmbeddedLens";

export default function LensPage() {
  const { accessToken } = useAuthorized();
  if (!accessToken) return null;
  return <EmbeddedLens />;
}
