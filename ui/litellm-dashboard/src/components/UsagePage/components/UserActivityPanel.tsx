import { Users } from "lucide-react";
import React from "react";

import { Panel } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";

import type { FetchUserPage } from "./userActivityData";
import { useUserActivityPage } from "./useUserActivityPage";
import UserActivityTable from "./UserActivityTable";

const PAGE_SIZE = 50;

interface UserActivityPanelProps {
  fetchUserPage: FetchUserPage;
}

const UserActivityPanel: React.FC<UserActivityPanelProps> = ({ fetchUserPage }) => {
  const page = useUserActivityPage(fetchUserPage, PAGE_SIZE);
  return (
    <Panel icon={Users} title="User activity" className="overflow-hidden" bodyClassName="px-0 pt-3 pb-0">
      <div className="px-4 pb-2">
        <UserActivityTable
          rows={page.rows}
          total={page.total}
          hasMore={page.hasMore}
          loading={page.loading}
          loadingMore={page.loadingMore}
          failed={page.failed}
          onLoadMore={page.loadMore}
          onRetryFirstPage={page.retryFirstPage}
          onRetryNextPage={page.retryNextPage}
        />
      </div>
    </Panel>
  );
};

export default UserActivityPanel;
