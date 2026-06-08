"use client";

import { SearchTools } from "@/components/SearchTools";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function SearchToolsRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <SearchTools accessToken={accessToken} userRole={userRole} userID={userId} />;
}
