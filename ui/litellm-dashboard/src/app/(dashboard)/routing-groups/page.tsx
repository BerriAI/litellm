"use client";

import RoutingGroupsView from "@/components/routing_groups/RoutingGroupsView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const RoutingGroupsPage = () => {
  const { accessToken, userId, userRole } = useAuthorized();

  return (
    <RoutingGroupsView
      accessToken={accessToken}
      userRole={userRole ?? ""}
      userId={userId ?? ""}
    />
  );
};

export default RoutingGroupsPage;
