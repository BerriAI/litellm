import { organizationKeys, useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useUserModels } from "@/app/(dashboard)/hooks/models/useModels";
import OrganizationFilters, { FilterState } from "@/app/(dashboard)/organizations/OrganizationFilters";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { organizationDeleteCall } from "@/components/networking";
import { OrgCreateDialog } from "@/components/organization/org-create/OrgCreateDialog";
import OrganizationInfoView from "@/components/organization/organization_view";
import { ORGANIZATION_TAB_URL_KEY, ORGANIZATION_TABS } from "@/components/organization/organizationTabs";
import { Button } from "@/components/ui/button";

import OrganizationsTable from "./OrganizationsTable";
import { organizationIdFilter, useOrganizationsTableState } from "./useOrganizationsTableState";

interface OrganizationsPanelProps {
  userRole: string;
  accessToken: string | null;
  premiumUser: boolean;
}

const ORGANIZATION_DETAIL_STATE = {
  org: parseAsString,
  tab: parseAsStringLiteral(ORGANIZATION_TABS),
};
const ORGANIZATION_DETAIL_URL_KEYS = { tab: ORGANIZATION_TAB_URL_KEY };

const OrganizationsPanel: React.FC<OrganizationsPanelProps> = ({ userRole, accessToken, premiumUser }) => {
  const { t } = useTranslation();
  const [{ org: selectedOrgId }, setOrganizationDetail] = useQueryStates(ORGANIZATION_DETAIL_STATE, {
    history: "push",
    urlKeys: ORGANIZATION_DETAIL_URL_KEYS,
  });
  const tableState = useOrganizationsTableState();
  const { setSearch, onColumnFiltersChange } = tableState;
  const filters: FilterState = { org_id: organizationIdFilter(tableState), org_alias: tableState.search };
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [orgToDelete, setOrgToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isOrgModalVisible, setIsOrgModalVisible] = useState(false);
  const [showFilters, setShowFilters] = useState(() => filters.org_id !== "");

  const queryClient = useQueryClient();
  const { data: organizations = [], isLoading } = useOrganizations({
    org_id: filters.org_id,
    org_alias: filters.org_alias,
  });
  const { data: userModels = [] } = useUserModels();

  const searchActive = Boolean(filters.org_id || filters.org_alias);

  const refetchOrganizations = () => queryClient.invalidateQueries({ queryKey: organizationKeys.lists() });

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    if (key === "org_alias") {
      setSearch(value);
      return;
    }
    onColumnFiltersChange(value ? [{ id: "org_id", value }] : []);
  };

  const handleFilterReset = () => {
    setSearch("");
    onColumnFiltersChange([]);
  };

  const handleDelete = (orgId: string | null) => {
    if (!orgId) return;

    setOrgToDelete(orgId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!orgToDelete || !accessToken) return;

    try {
      setIsDeleting(true);
      await organizationDeleteCall(accessToken, orgToDelete);
      toast.success(t("organizations.panel.toasts.deleted"));

      setIsDeleteModalOpen(false);
      setOrgToDelete(null);
      await refetchOrganizations();
    } catch (error) {
      console.error("Error deleting organization:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setOrgToDelete(null);
  };

  if (!premiumUser) {
    return (
      <div className="mx-4 mt-4">
        <p className="text-sm text-muted-foreground">
          <Trans
            i18nKey="organizations.panel.enterpriseNotice"
            components={{
              pricing: (
                <a
                  href="https://www.litellm.ai/#pricing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline-offset-4 hover:underline"
                />
              ),
            }}
          />
        </p>
      </div>
    );
  }

  return (
    <div className="mx-4 mt-4 flex flex-col gap-4">
      {(userRole === "Admin" || userRole === "Org Admin") && (
        <Button className="w-fit" onClick={() => setIsOrgModalVisible(true)}>
          {t("organizations.panel.createButton")}
        </Button>
      )}

      {selectedOrgId ? (
        <OrganizationInfoView
          organizationId={selectedOrgId}
          onClose={() => void setOrganizationDetail(null)}
          accessToken={accessToken}
          is_org_admin={true}
          is_proxy_admin={userRole === "Admin"}
          userModels={userModels}
        />
      ) : (
        <>
          <p className="text-sm text-muted-foreground">{t("organizations.panel.listHint")}</p>
          <OrganizationFilters
            filters={filters}
            showFilters={showFilters}
            onToggleFilters={setShowFilters}
            onChange={handleFilterChange}
            onReset={handleFilterReset}
          />
          <OrganizationsTable
            organizations={organizations}
            isLoading={isLoading}
            userRole={userRole}
            searchActive={searchActive}
            onOrganizationClick={(organizationId) => void setOrganizationDetail({ org: organizationId, tab: null })}
            onEditClick={(organizationId) => void setOrganizationDetail({ org: organizationId, tab: "settings" })}
            onDeleteClick={handleDelete}
          />
        </>
      )}

      <OrgCreateDialog open={isOrgModalVisible} onOpenChange={setIsOrgModalVisible} accessToken={accessToken || ""} />

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={t("organizations.panel.deleteModal.title")}
        message={t("organizations.panel.deleteModal.message")}
        resourceInformationTitle={t("organizations.panel.deleteModal.infoTitle")}
        resourceInformation={[{ label: t("organizations.panel.deleteModal.idLabel"), value: orgToDelete, code: true }]}
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeleting}
      />
    </div>
  );
};

export default OrganizationsPanel;
