import { teamListCall, organizationListCall, keyAliasesCall } from "../networking"
import { Team } from "./key_list";
import { Organization } from "../networking";

/**
 * Fetches all key aliases via the dedicated /key/aliases endpoint
 * @param accessToken The access token for API authentication
 * @returns Array of all unique key aliases
 */
export const fetchAllKeyAliases = async (accessToken: string | null): Promise<string[]> => {
  if (!accessToken) return [];

  try {
    const { aliases } = await keyAliasesCall(accessToken as unknown as string);
    // Defensive dedupe & null-guard
    return Array.from(new Set((aliases || []).filter(Boolean)));
  } catch (error) {
    console.error("Error fetching all key aliases:", error);
    return [];
  }
};


/**
 * Fetches all teams across all pages
 * @param accessToken The access token for API authentication
 * @param organizationId Optional organization ID to filter teams
 * @returns Array of all teams
 */
export const fetchAllTeams = async (accessToken: string | null, organizationId?: string | null): Promise<Team[]> => {
  if (!accessToken) return [];

  try {
    let allTeams: Team[] = [];
    let currentPage = 1;
    let hasMorePages = true;

    while (hasMorePages) {
      const response = await teamListCall(accessToken, organizationId || null, null);

      // Add teams from this page
      allTeams = [...allTeams, ...response];

      // Check if there are more pages
      if (currentPage < response.total_pages) {
        currentPage++;
      } else {
        hasMorePages = false;
      }
    }

    return allTeams;
  } catch (error) {
    console.error("Error fetching all teams:", error);
    return [];
  }
};

/**
 * Fetches all organizations across all pages
 * @param accessToken The access token for API authentication
 * @returns Array of all organizations
 */
export const fetchAllOrganizations = async (accessToken: string | null): Promise<Organization[]> => {
  if (!accessToken) return [];

  try {
    let allOrganizations: Organization[] = [];
    let currentPage = 1;
    let hasMorePages = true;

    while (hasMorePages) {
      const response = await organizationListCall(accessToken);

      // Add organizations from this page
      allOrganizations = [...allOrganizations, ...response];

      // Check if there are more pages
      if (currentPage < response.total_pages) {
        currentPage++;
      } else {
        hasMorePages = false;
      }
    }

    return allOrganizations;
  } catch (error) {
    console.error("Error fetching all organizations:", error);
    return [];
  }
};
