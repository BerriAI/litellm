"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Fallbacks from "@/components/Settings/RouterSettings/Fallbacks/Fallbacks";
import { all_admin_roles } from "@/utils/roles";

export default function FallbacksPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  if (!userRole || !all_admin_roles.includes(userRole)) return null;

  return (
    <div className="w-full space-y-6 px-8 py-6">
      <div>
        <h1 className="text-xl font-semibold">Fallbacks</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Choose which models to try, in order, when a primary model fails.
        </p>
      </div>
      <Fallbacks accessToken={accessToken} userRole={userRole} userID={userId} />
    </div>
  );
}
