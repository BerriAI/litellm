"use client";

import PromptsPanel from "./components";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Prompts() {
  const { accessToken, userRole } = useAuthorized();
  return (
    <>
      <DeprecationBanner featureName="Prompt Management" />
      <PromptsPanel accessToken={accessToken} userRole={userRole} />
    </>
  );
}
