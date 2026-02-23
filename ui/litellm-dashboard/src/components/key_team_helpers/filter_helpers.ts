import { teamListCall, organizationListCall, keyAliasesCall, keyListCall } from "../networking";
import { Team } from "./key_list";
import { Organization } from "../networking";

export interface TeamFilterOptions {
  keyAliases: string[];
  organizationIds: string[];
  userIds: Array<{ id: string; email: string }>;
}

/**
 * Fetches filter options (key aliases, org IDs, user IDs) scoped to a team's keys.
 * Used by TeamVirtualKeysTable to show only relevant filter options.
 */
export const fetchTeamFilterOptions = async (
  accessToken: string | null,
  teamId: string,
): Promise<TeamFilterOptions> => {
  if (!accessToken || !teamId) {
    return { keyAliases: [], organizationIds: [], userIds: [] };
  }

  try {
    const keyAliases = new Set<string>();
    const organizationIds = new Set<string>();
    const userMap = new Map<string, string>(); // user_id -> user_email

    let page = 1;
    let totalPages = 1;

    do {
      const response = await keyListCall(
        accessToken,
        null,
        teamId,
        null,
        null,
        null,
        page,
        100,
        null,
        null,
        "user",
        null,
      );

      const keys = response?.keys || [];
      totalPages = response?.total_pages ?? 1;

      for (const key of keys) {
        const alias = key?.key_alias;
        if (alias && typeof alias === "string") {
          keyAliases.add(alias.trim());
        }
        const orgId = key?.organization_id;
        if (orgId && typeof orgId === "string") {
          organizationIds.add(orgId.trim());
        }
        const userId = key?.user_id;
        if (userId && typeof userId === "string") {
          const email = key?.user?.user_email || userId;
          userMap.set(userId, email);
        }
      }

      page++;
    } while (page <= totalPages);

    return {
      keyAliases: Array.from(keyAliases).sort(),
      organizationIds: Array.from(organizationIds).sort(),
      userIds: Array.from(userMap.entries()).map(([id, email]) => ({ id, email })),
    };
  } catch (error) {
    console.error("Error fetching team filter options:", error);
    return { keyAliases: [], organizationIds: [], userIds: [] };
  }
};

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
