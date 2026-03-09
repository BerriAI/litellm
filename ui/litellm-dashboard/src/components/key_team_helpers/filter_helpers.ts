import { teamListCall, organizationListCall, keyListCall } from "../networking";
import { Team } from "./key_list";
import { Organization } from "../networking";

export interface TeamFilterOptions {
  keyAliases: string[];
  organizationIds: string[];
  userIds: Array<{ id: string; email: string }>;
}

const FILTER_OPTIONS_PAGE_SIZE = 100; // API max per page
const MAX_PAGES = 10; // Cap at 1000 keys; filter completeness beyond ~500 has diminishing returns

const processKeysIntoOptions = (
  keys: Array<Record<string, unknown>>,
  keyAliases: Set<string>,
  organizationIds: Set<string>,
  userMap: Map<string, string>,
) => {
  for (const key of keys) {
    const alias = key?.key_alias;
    if (alias && typeof alias === "string") {
      keyAliases.add(alias.trim());
    }
    const orgId = key?.organization_id ?? key?.org_id;
    if (orgId && typeof orgId === "string") {
      organizationIds.add(orgId.trim());
    }
    const userId = key?.user_id;
    if (userId && typeof userId === "string") {
      const email = (key?.user as { user_email?: string })?.user_email || userId;
      userMap.set(userId, email);
    }
  }
};

/**
 * Fetches filter options (key aliases, org IDs, user IDs) from team keys.
 * Fetches page 1 first to get totalPages, then batches remaining pages with
 * Promise.allSettled (preserves successful pages if some fail). Capped at 10 pages (1000 keys)
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
    const userMap = new Map<string, string>();

    // First request: get page 1 and totalPages
    const firstResponse = await keyListCall(
      accessToken,
      null,
      teamId,
      null,
      null,
      null,
      1,
      FILTER_OPTIONS_PAGE_SIZE,
      null,
      null,
      "user",
      null,
    );

    const firstKeys = firstResponse?.keys || [];
    const totalPages = firstResponse?.total_pages ?? 1;
    processKeysIntoOptions(firstKeys, keyAliases, organizationIds, userMap);

    // Batch fetch remaining pages (2 through min(totalPages, MAX_PAGES)) in parallel
    const pagesToFetch = Math.min(totalPages, MAX_PAGES) - 1;
    if (pagesToFetch > 0) {
      const pagePromises = Array.from({ length: pagesToFetch }, (_, i) =>
        keyListCall(
          accessToken,
          null,
          teamId,
          null,
          null,
          null,
          i + 2,
          FILTER_OPTIONS_PAGE_SIZE,
          null,
          null,
          "user",
          null,
        ),
      );
      const results = await Promise.allSettled(pagePromises);
      for (const result of results) {
        if (result.status === "fulfilled") {
          processKeysIntoOptions(result.value?.keys || [], keyAliases, organizationIds, userMap);
        }
      }
    }

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
