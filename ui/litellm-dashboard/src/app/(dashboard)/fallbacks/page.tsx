"use client";

import Fallbacks from "@/components/Settings/RouterSettings/Fallbacks/Fallbacks";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function FallbacksPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return (
    <div className="w-full px-8 py-6">
      <Fallbacks accessToken={accessToken} userRole={userRole} userID={userId} />
    </div>
  );
}
