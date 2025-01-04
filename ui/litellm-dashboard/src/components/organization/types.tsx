export interface Organization {
    organization_id: string;
    organization_name: string;
    spend: number;
    max_budget: number | null;
    models: string[];
    tpm_limit: number | null;
    rpm_limit: number | null;
    // Incorporating the info directly into the organization
    keys: Array<{ id: string }>;
    members_with_roles: Array<{ id: string }>;
  }