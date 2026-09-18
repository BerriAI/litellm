"use client";

import { SearchTools } from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function SearchToolsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <SearchTools accessToken={accessToken} userRole={userRole} userID={userId} />;
}
