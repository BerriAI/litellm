"use client";

import PromptsPanel from "@/components/prompts";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const PromptsPage = () => {
  const { accessToken } = useAuthorized();

  return <PromptsPanel accessToken={accessToken} />;
};

export default PromptsPage;
