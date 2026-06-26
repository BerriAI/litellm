"use client";
import { useState } from "react";
import { Alert } from "antd";
import { useTranslation } from "react-i18next";
import { useDeletedKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedKeysTable } from "./DeletedKeysTable/DeletedKeysTable";

export default function DeletedKeysPage() {
  const { t } = useTranslation();
  const { premiumUser } = useAuthorized();
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize] = useState(50);

  const { data: keysData, isPending: isLoading, isFetching } = useDeletedKeys(pageIndex + 1, pageSize);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert
          type="info"
          banner
          showIcon
          message={t("deletedKeys.deletedKeysPage.comingSoonToEnterprise")}
          description={t("deletedKeys.deletedKeysPage.auditingDescription")}
        />
      )}
      <DeletedKeysTable
        keys={keysData?.keys || []}
        totalCount={keysData?.total_count || 0}
        isLoading={isLoading}
        isFetching={isFetching}
        pageIndex={pageIndex}
        pageSize={pageSize}
        onPageChange={setPageIndex}
      />
    </div>
  );
}
