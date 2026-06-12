"use client";

import { SearchTools } from "@/components/SearchTools";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function SearchToolsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <SearchTools accessToken={accessToken} userRole={userRole} userID={userId} />;
}
