import { organizationKeys, useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useUserModels } from "@/app/(dashboard)/hooks/models/useModels";
import OrganizationFilters, { FilterState } from "@/app/(dashboard)/organizations/OrganizationFilters";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Building2 } from "lucide-react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { organizationDeleteCall } from "@/components/networking";
import { OrgCreateDialog } from "@/components/organization/org-create/OrgCreateDialog";
import OrganizationInfoView from "@/components/organization/organization_view";
import { ORGANIZATION_TAB_URL_KEY, ORGANIZATION_TABS } from "@/components/organization/organizationTabs";
import { Button } from "@/components/ui/button";
import { PageContent } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";

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
      <PageContent>
        <PageHeader>
          <PageHeaderTitle>
            <Building2 />
            {t("shell.nav.items.organizations")}
          </PageHeaderTitle>
          <PageHeaderDescription>
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
          </PageHeaderDescription>
        </PageHeader>
      </PageContent>
    );
  }

  return (
    <>
      <PageContent>
        <PageHeader>
          <PageHeaderTitle>
            <Building2 />
            {t("shell.nav.items.organizations")}
          </PageHeaderTitle>
          <PageHeaderDescription>{t("organizations.panel.listHint")}</PageHeaderDescription>
          {(userRole === "Admin" || userRole === "Org Admin") && (
            <PageHeaderControls>
              <Button onClick={() => setIsOrgModalVisible(true)}>{t("organizations.panel.createButton")}</Button>
            </PageHeaderControls>
          )}
        </PageHeader>
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
      </PageContent>

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
    </>
  );
};

export default OrganizationsPanel;
