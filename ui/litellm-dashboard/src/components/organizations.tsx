import React, { useState, useEffect } from "react";
import { organizationListCall, organizationMemberAddCall, Member, modelAvailableCall } from "./networking";
import {
  Col,
  Grid,
  Text
} from "@tremor/react";
import OrganizationForm from "@/components/organization/add_org";
import AddOrgAdmin from "@/components/organization/add_org_admin";
import MemberListTable from "@/components/organization/view_members_of_org";
import { Select, SelectItem } from "@tremor/react";
const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}
interface TeamProps {
  teams: any[] | null;
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
  userID: string | null;
  userRole: string | null;
  premiumUser: boolean;
}


import DataTable from "@/components/common_components/all_view";
import { Action } from "@/components/common_components/all_view";
import { Typography, message } from "antd";
import { Organization, EditModalProps } from "@/components/organization/types";

interface OrganizationsTableProps {
  organizations: Organization[];
  userRole?: string;
  onEdit?: (organization: Organization) => void;
  onDelete?: (organization: Organization) => void;
  isDeleteModalOpen: boolean;
  setIsDeleteModalOpen: (value: boolean) => void;
  selectedOrganization: Organization | null;
  setSelectedOrganization: (value: Organization | null) => void;
}


const EditOrganizationModal: React.FC<EditModalProps> = ({
  visible,
  onCancel,
  entity,
  onSubmit
}) => {
  return <div/>;
};


// Inside your Teams component
const OrganizationsTable: React.FC<OrganizationsTableProps> = ({
  organizations,
  userRole,
  onEdit,
  onDelete,
  isDeleteModalOpen,
  setIsDeleteModalOpen,
  selectedOrganization,
  setSelectedOrganization
}) => {
  
  const columns = [
    {
      header: "Organization Name",
      accessor: "organization_alias",
      width: "4px",
      style: {
        whiteSpace: "pre-wrap",
        overflow: "hidden"
      }
    },
    {
      header: "Organization ID",
      accessor: "organization_id",
      width: "4px",
      style: {
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        fontSize: "0.75em"
      }
    },
    {
      header: "Spend (USD)",
      accessor: "spend"
    },
    {
      header: "Budget (USD)",
      accessor: "max_budget",
      cellRenderer: (value: number | null) => 
        value !== null && value !== undefined ? value : "No limit"
    },
    {
      header: "Models",
      accessor: "models"
    },
    {
      header: "TPM / RPM Limits",
      accessor: "limits",
      cellRenderer: (value: any, row: Organization) => (
        <div className="text-sm">
          <span>TPM: {row.tpm_limit ? row.tpm_limit : "Unlimited"}</span>
          <br />
          <span>RPM: {row.rpm_limit ? row.rpm_limit : "Unlimited"}</span>
        </div>
      )
    },
    {
      header: "Info",
      accessor: "info",
      cellRenderer: (value: any, row: Organization) => (
        <div className="space-y-1">
          <div className="text-sm">
            {row.members?.length || 0} Members
          </div>
        </div>
      )
    }
  ];

  const actions: Action[] = [
    ...(onEdit && userRole === "Admin" ? [{
      icon: undefined, // Replace with your PencilAltIcon
      onClick: (org: Organization) => onEdit(org),
      condition: () => userRole === "Admin",
      tooltip: "Edit organization"
    }] : []),
    ...(onDelete && userRole === "Admin" ? [{
      icon: undefined, // Replace with your TrashIcon
      onClick: (org: Organization) => onDelete(org),
      condition: () => userRole === "Admin",
      tooltip: "Delete organization"
    }] : [])
  ];

  return (
    <DataTable
      data={organizations}
      columns={columns}
      actions={actions}
      emptyMessage="No organizations available"
      deleteModal={{
        isOpen: isDeleteModalOpen,
        onConfirm: () => {
          if (selectedOrganization && onDelete) {
            onDelete(selectedOrganization);
          }
          setIsDeleteModalOpen(false);
          setSelectedOrganization(null);
        },
        onCancel: () => {
          setIsDeleteModalOpen(false);
          setSelectedOrganization(null);
        },
        title: "Delete Organization",
        message: "Are you sure you want to delete this organization?"
      }}
    />
  );
};

const Organizations: React.FC<TeamProps> = ({
  accessToken,
  userID,
  userRole,
  premiumUser
}) => {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const { Title, Paragraph } = Typography;
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedOrganization, setSelectedOrganization] = useState<Organization | null>(null);
  const [userModels, setUserModels] = useState([]);

  useEffect(() => {
    if (!accessToken || !userID || !userRole) return;

    const fetchUserModels = async () => {
      try {
        const model_available = await modelAvailableCall(
          accessToken,
          userID,
          userRole
        );
        let available_model_names = model_available["data"].map(
          (element: { id: string }) => element.id
        );
        console.log("available_model_names:", available_model_names);
        setUserModels(available_model_names);
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    const fetchData = async () => {
      let givenOrganizations;
      givenOrganizations = await organizationListCall(accessToken)
      console.log(`givenOrganizations: ${givenOrganizations}`)
      setOrganizations(givenOrganizations)
      sessionStorage.setItem('organizations', JSON.stringify(givenOrganizations));
    }
    if (premiumUser) {
      fetchUserModels()
      fetchData()
    }
  }, [accessToken]);

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    if (!selectedOrganization || !accessToken) return;
    try {
      let member: Member = {
        user_email: formValues.user_email,
        user_id: formValues.user_id,
        role: formValues.role
      }
      await organizationMemberAddCall(
        accessToken,
        selectedOrganization["organization_id"],
        member
      );
      message.success("Member added");
    } catch (error) {
      console.error("Error creating the team:", error);
      message.error("Error creating the organization: " + error);
    }
};

  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Col numColSpan={1}>
          <Title level={4}>✨ All Organizations</Title>
          <Text className="mb-2">This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://www.litellm.ai/#trial" className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">here</a></Text>
          {userRole ? OrganizationsTable({organizations, userRole, isDeleteModalOpen, setIsDeleteModalOpen, selectedOrganization, setSelectedOrganization}) : null}
        </Col>
        {userRole == "Admin" && accessToken && premiumUser ? <OrganizationForm
          title="Organization"
          accessToken={accessToken}
          availableModels={userModels}
          submitButtonText="Create Organization"
        /> : null}
        {premiumUser ? 
        <Col numColSpan={1}>  
        <Title level={4}>Organization Members</Title>
        <Paragraph>
            If you belong to multiple organizations, this setting controls which organizations&apos;
            members you see.
        </Paragraph>

          {organizations && organizations.length > 0 ? (
            <Select>
              {organizations.map((organization: any, index) => (
                <SelectItem
                  key={index}
                  value={String(index)}
                  onClick={() => {
                    setSelectedOrganization(organization);
                  }}
                >
                  {organization["organization_alias"]}
                </SelectItem>
              ))}
            </Select>
          ) : (
            <Paragraph>
              No team created. <b>Defaulting to personal account.</b>
            </Paragraph>
          )}
        </Col> : null}
        {userRole == "Admin" && userID && selectedOrganization && premiumUser ? <AddOrgAdmin userRole={userRole} userID={userID} selectedOrganization={selectedOrganization} onMemberAdd={handleMemberCreate} /> : null}
        {userRole == "Admin" && userID && selectedOrganization && premiumUser ? <MemberListTable  selectedEntity={selectedOrganization} onEditSubmit={() => {}} editModalComponent={EditOrganizationModal} entityType="organization" /> : null}
      </Grid>
    </div>
  );
};

export default Organizations;
