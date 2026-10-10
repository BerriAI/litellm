"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Fallbacks from "@/components/Settings/RouterSettings/Fallbacks/Fallbacks";

export default function FallbacksPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold">Fallbacks</h1>
        <p className="text-sm text-muted-foreground">
          Automatically retry a different model when the primary fails
        </p>
      </div>
      <Fallbacks accessToken={accessToken} userRole={userRole} userID={userId} />
    </div>
  );
}
