import { organizationKeys, useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useUserModels } from "@/app/(dashboard)/hooks/models/useModels";
import OrganizationFilters, { FilterState } from "@/app/(dashboard)/organizations/OrganizationFilters";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import React, { useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { organizationDeleteCall } from "@/components/networking";
import { OrgCreateDialog } from "@/components/organization/org-create/OrgCreateDialog";
import OrganizationInfoView from "@/components/organization/organization_view";
import { Button } from "@/components/ui/button";

import OrganizationsTable from "./OrganizationsTable";

interface OrganizationsPanelProps {
  userRole: string;
  accessToken: string | null;
  premiumUser: boolean;
}

const OrganizationsPanel: React.FC<OrganizationsPanelProps> = ({ userRole, accessToken, premiumUser }) => {
  const [selectedOrgId, setSelectedOrgId] = useQueryState("org", parseAsString.withOptions({ history: "push" }));
  const [editOrg, setEditOrg] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [orgToDelete, setOrgToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isOrgModalVisible, setIsOrgModalVisible] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({ org_id: "", org_alias: "" });

  const queryClient = useQueryClient();
  const { data: organizations = [], isLoading } = useOrganizations({
    org_id: filters.org_id,
    org_alias: filters.org_alias,
  });
  const { data: userModels = [] } = useUserModels();

  const searchActive = Boolean(filters.org_id || filters.org_alias);

  const refetchOrganizations = () => queryClient.invalidateQueries({ queryKey: organizationKeys.lists() });

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    setFilters((previousFilters) => ({ ...previousFilters, [key]: value }));
  };

  const handleFilterReset = () => {
    setFilters({ org_id: "", org_alias: "" });
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
      toast.success("Organization deleted successfully");

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
          This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key{" "}
          <a
            href="https://www.litellm.ai/#pricing"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline-offset-4 hover:underline"
          >
            here
          </a>
          .
        </p>
      </div>
    );
  }

  return (
    <div className="mx-4 mt-4 flex flex-col gap-4">
      {(userRole === "Admin" || userRole === "Org Admin") && (
        <Button className="w-fit" onClick={() => setIsOrgModalVisible(true)}>
          + Create New Organization
        </Button>
      )}

      {selectedOrgId ? (
        <OrganizationInfoView
          organizationId={selectedOrgId}
          onClose={() => {
            void setSelectedOrgId(null);
            setEditOrg(false);
          }}
          accessToken={accessToken}
          is_org_admin={true}
          is_proxy_admin={userRole === "Admin"}
          userModels={userModels}
          editOrg={editOrg}
        />
      ) : (
        <>
          <p className="text-sm text-muted-foreground">Click on an organization ID to view its details.</p>
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
            onOrganizationClick={(organizationId) => {
              setEditOrg(false);
              void setSelectedOrgId(organizationId);
            }}
            onEditClick={(organizationId) => {
              void setSelectedOrgId(organizationId);
              setEditOrg(true);
            }}
            onDeleteClick={handleDelete}
          />
        </>
      )}

      <OrgCreateDialog open={isOrgModalVisible} onOpenChange={setIsOrgModalVisible} accessToken={accessToken || ""} />

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Organization?"
        message="Are you sure you want to delete this organization? This action cannot be undone."
        resourceInformationTitle="Organization Information"
        resourceInformation={[{ label: "Organization ID", value: orgToDelete, code: true }]}
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeleting}
      />
    </div>
  );
};

export default OrganizationsPanel;
