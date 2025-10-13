export interface EditModalProps {
  visible: boolean;
  onCancel: () => void;
  entity: Organization;
  onSubmit: (entity: Organization) => void;
}

export interface OrganizationMember {
  user_id: string;
  user_role: string;
}

export interface Organization {
  organization_id: string;
  organization_name: string;
  spend: number;
  max_budget: number | null;
  models: string[];
  tpm_limit: number | null;
  rpm_limit: number | null;
  members: OrganizationMember[] | null;
}
