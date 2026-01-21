import { useCallback, useEffect, useState, useRef } from "react";
import { KeyResponse } from "../key_team_helpers/key_list";
import { keyListCall, Organization } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllOrganizations, fetchAllTeams } from "./filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export interface FilterState {
  "Team ID": string;
  "Organization ID": string;
  "Key Alias": string;
  [key: string]: string;
  "User ID": string;
  "Sort By": string;
  "Sort Order": string;
}

export function useFilterLogic({
  keys,
  teams,
  organizations,
  pageSize,
}: {
  keys: KeyResponse[];
  teams: Team[] | null;
  organizations: Organization[] | null;
  pageSize?: number;
}) {
  const defaultFilters: FilterState = {
    "Team ID": "",
    "Organization ID": "",
    "Key Alias": "",
    "User ID": "",
    "Sort By": "created_at",
    "Sort Order": "desc",
  };
  const { accessToken } = useAuthorized();
  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  const [allTeams, setAllTeams] = useState<Team[]>(teams || []);
  const [allOrganizations, setAllOrganizations] = useState<Organization[]>(organizations || []);
  const [filteredKeys, setFilteredKeys] = useState<KeyResponse[]>(keys);
  const lastSearchTimestamp = useRef(0);
  const isServerSideFilterActive = useRef(false);

  const debouncedSearch = useCallback(
    debounce(async (filters: FilterState) => {
      if (!accessToken) {
        return;
      }

      const currentTimestamp = Date.now();
      lastSearchTimestamp.current = currentTimestamp;

      try {
        // Make the API call using keyListCall with all filter parameters
        const data = await keyListCall(
          accessToken,
          filters["Organization ID"] || null,
          filters["Team ID"] || null,
          filters["Key Alias"] || null,
          filters["User ID"] || null,
          filters["Key Hash"] || null,
          1, // Reset to first page when searching
          pageSize ?? defaultPageSize,
          filters["Sort By"] || null,
          filters["Sort Order"] || null,
        );

        // Only update state if this is the most recent search
        if (currentTimestamp === lastSearchTimestamp.current) {
          if (data) {
            setFilteredKeys(data.keys);
          }
        }
      } catch (error) {
        console.error("Error searching keys:", error);
      }
    }, 300),
    [accessToken, pageSize],
  );

  // Apply filters to keys whenever keys or filters change
  // BUT only if we're not using server-side filtering (sorting, key alias search, etc.)
  useEffect(() => {
    // If server-side filtering is active, don't overwrite the results from debouncedSearch
    // The server already returned the filtered/sorted data
    if (isServerSideFilterActive.current) {
      return;
    }

    if (!keys) {
      setFilteredKeys([]);
      return;
    }

    let result = [...keys];

    // Apply Team ID filter (client-side, only when not using server-side filters)
    if (filters["Team ID"]) {
      result = result.filter((key) => key.team_id === filters["Team ID"]);
    }

    // Apply Organization ID filter (client-side, only when not using server-side filters)
    if (filters["Organization ID"]) {
      result = result.filter((key) => key.organization_id === filters["Organization ID"]);
    }

    setFilteredKeys(result);
  }, [keys, filters]);

  // Fetch all data for filters when component mounts
  useEffect(() => {
    const loadAllFilterData = async () => {
      // Load all teams - no organization filter needed here
      const teamsData = await fetchAllTeams(accessToken);
      if (teamsData.length > 0) {
        setAllTeams(teamsData);
      }

      // Load all organizations
      const orgsData = await fetchAllOrganizations(accessToken);
      if (orgsData.length > 0) {
        setAllOrganizations(orgsData);
      }
    };

    if (accessToken) {
      loadAllFilterData();
    }
  }, [accessToken]);

  const queryAllKeysQuery = useQuery({
    queryKey: ["allKeys"],
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required");
      return await fetchAllKeyAliases(accessToken);
    },
    enabled: !!accessToken,
  });
  const allKeyAliases = queryAllKeysQuery.data || [];

  // Update teams and organizations when props change
  useEffect(() => {
    if (teams && teams.length > 0) {
      setAllTeams((prevTeams) => {
        // Only update if we don't already have a larger set of teams
        return prevTeams.length < teams.length ? teams : prevTeams;
      });
    }
  }, [teams]);

  useEffect(() => {
    if (organizations && organizations.length > 0) {
      setAllOrganizations((prevOrgs) => {
        // Only update if we don't already have a larger set of organizations
        return prevOrgs.length < organizations.length ? organizations : prevOrgs;
      });
    }
  }, [organizations]);

  const handleFilterChange = (newFilters: Record<string, string>) => {
    // Check if any server-side filter will be active after this change
    // This must be set BEFORE setFilters to prevent the useEffect from overwriting
    const hasServerSideFilter =
      (newFilters["Sort By"] && newFilters["Sort By"] !== "created_at") ||
      (newFilters["Sort Order"] && newFilters["Sort Order"] !== "desc") ||
      newFilters["Key Alias"] ||
      newFilters["User ID"] ||
      newFilters["Key Hash"];

    isServerSideFilterActive.current = !!hasServerSideFilter;

    const newFiltersState = {
      "Team ID": newFilters["Team ID"] || "",
      "Organization ID": newFilters["Organization ID"] || "",
      "Key Alias": newFilters["Key Alias"] || "",
      "User ID": newFilters["User ID"] || "",
      "Sort By": newFilters["Sort By"] || "created_at",
      "Sort Order": newFilters["Sort Order"] || "desc",
    };

    // Update filters state
    setFilters(newFiltersState);

    // Fetch keys based on new filters
    const updatedFilters = {
      ...filters,
      ...newFilters,
    };
    debouncedSearch(updatedFilters);
  };

  const handleFilterReset = () => {
    // Reset the server-side filter flag so client-side filtering can take over
    isServerSideFilterActive.current = false;

    // Reset filters state
    setFilters(defaultFilters);

    // Reset selections
    debouncedSearch(defaultFilters);
  };

  return {
    filters,
    filteredKeys,
    allKeyAliases,
    allTeams,
    allOrganizations,
    handleFilterChange,
    handleFilterReset,
  };
}
