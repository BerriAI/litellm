import React, { useState, useEffect } from "react";
import { organizationListCall, organizationMemberAddCall, Member } from "./networking";
import { Title } from "@tremor/react";
import {
  Col,
  Grid,
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
import OrganizationForm from "@/components/organization/add_org";
import AddOrgAdmin from "@/components/organization/add_org_admin";
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
}

interface EditTeamModalProps {
  visible: boolean;
  onCancel: () => void;
  team: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

import {
  teamCreateCall,
  teamMemberAddCall,
  Member,
  modelAvailableCall,
  teamListCall
} from "./networking";

import DataTable from "@/components/common_components/all_view";
import { Action } from "@/components/common_components/all_view";
import { Typography, message } from "antd";
import { Organization } from "@/components/organization/types";

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
            {row.keys?.length || 0} Keys
          </div>
          <div className="text-sm">
            {row.members_with_roles?.length || 0} Members
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
}) => {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const { Title, Paragraph } = Typography;
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedOrganization, setSelectedOrganization] = useState<Organization | null>(null);


  useEffect(() => {
    if (!accessToken) return;

    const storedOrganizations = sessionStorage.getItem('organizations');
    if (storedOrganizations) {
      setOrganizations(JSON.parse(storedOrganizations));
    } else {
      const fetchData = async () => {
        let givenOrganizations;
        givenOrganizations = await organizationListCall(accessToken)
        console.log(`givenOrganizations: ${givenOrganizations}`)
        setOrganizations(givenOrganizations)
        sessionStorage.setItem('organizations', JSON.stringify(givenOrganizations));
      }
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
    }
};

  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Col numColSpan={1}>
          <Title level={4}>All Organizations</Title>
          {userRole ? OrganizationsTable({organizations, userRole, isDeleteModalOpen, setIsDeleteModalOpen, selectedOrganization, setSelectedOrganization}) : null}
        </Col>
        {userRole == "Admin" && accessToken ? <OrganizationForm
          title="Organization"
          accessToken={accessToken}
          availableModels={['model1', 'model2', 'model3']}
          submitButtonText="Create Organization"
        /> : null}
        <Col numColSpan={1}>  
        <Title level={4}>Organization Members</Title>
          <Paragraph>
            If you belong to multiple organizations, this setting controls which organizations'
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
        </Col>
        {userRole == "Admin" && userID && selectedOrganization ? <AddOrgAdmin userRole={userRole} userID={userID} selectedOrganization={selectedOrganization} onMemberAdd={handleMemberCreate} /> : null}
      </Grid>
    </div>
  );
};

export default Organizations;
