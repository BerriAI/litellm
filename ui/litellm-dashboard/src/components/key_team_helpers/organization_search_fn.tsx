import { Organization } from "../networking";

export const createOrgSearchFunction = (organizations: Organization[] | null) => {
  return async (searchText: string): Promise<Array<{ label: string; value: string }>> => {
    if (!organizations || !searchText.trim()) {
      return [];
    }

    // Find organizations that match the search text by alias
    const matchingOrgs: Array<{ label: string; value: string }> = [];

    organizations.forEach((org) => {
      if (org.organization_alias && org.organization_alias.toLowerCase().includes(searchText.toLowerCase())) {
        matchingOrgs.push({
          label: `${org.organization_alias} (${org.organization_id})`,
          value: org.organization_id || "",
        });
      }
    });

    return matchingOrgs;
  };
};
