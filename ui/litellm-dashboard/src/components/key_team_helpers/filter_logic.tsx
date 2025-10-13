import { useCallback, useEffect, useState, useRef } from "react";
import { KeyResponse } from "../key_team_helpers/key_list";
import { keyListCall, Organization } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllOrganizations, fetchAllTeams } from "./filter_helpers";
import { debounce } from "lodash";
import { defaultPageSize } from "../constants";

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
  accessToken,
}: {
  keys: KeyResponse[];
  teams: Team[] | null;
  organizations: Organization[] | null;
  accessToken: string | null;
}) {
  const defaultFilters: FilterState = {
    "Team ID": "",
    "Organization ID": "",
    "Key Alias": "",
    "User ID": "",
    "Sort By": "created_at",
    "Sort Order": "desc",
  };
  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  const [allTeams, setAllTeams] = useState<Team[]>(teams || []);
  const [allOrganizations, setAllOrganizations] = useState<Organization[]>(organizations || []);
  const [filteredKeys, setFilteredKeys] = useState<KeyResponse[]>(keys);
  const lastSearchTimestamp = useRef(0);
  const debouncedSearch = useCallback(
    debounce(async (filters: FilterState) => {
      if (!accessToken) {
        return;
      }

      const currentTimestamp = Date.now();
      lastSearchTimestamp.current = currentTimestamp;

      try {
        // Make the API call using userListCall with all filter parameters
        const data = await keyListCall(
          accessToken,
          filters["Organization ID"] || null,
          filters["Team ID"] || null,
          filters["Key Alias"] || null,
          filters["User ID"] || null,
          filters["Key Hash"] || null,
          1, // Reset to first page when searching
          defaultPageSize,
          filters["Sort By"] || null,
          filters["Sort Order"] || null,
        );

        // Only update state if this is the most recent search
        if (currentTimestamp === lastSearchTimestamp.current) {
          if (data) {
            setFilteredKeys(data.keys);
            console.log("called from debouncedSearch filters:", JSON.stringify(filters));
            console.log("called from debouncedSearch data:", JSON.stringify(data));
          }
        }
      } catch (error) {
        console.error("Error searching users:", error);
      }
    }, 300),
    [accessToken],
  );
  // Apply filters to keys whenever keys or filters change
  useEffect(() => {
    if (!keys) {
      setFilteredKeys([]);
      return;
    }

    let result = [...keys];

    // Apply Team ID filter
    if (filters["Team ID"]) {
      result = result.filter((key) => key.team_id === filters["Team ID"]);
    }

    // Apply Organization ID filter
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
    // Update filters state
    setFilters({
      "Team ID": newFilters["Team ID"] || "",
      "Organization ID": newFilters["Organization ID"] || "",
      "Key Alias": newFilters["Key Alias"] || "",
      "User ID": newFilters["User ID"] || "",
      "Sort By": newFilters["Sort By"] || "created_at",
      "Sort Order": newFilters["Sort Order"] || "desc",
    });

    // Fetch keys based on new filters
    const updatedFilters = {
      ...filters,
      ...newFilters,
    };
    debouncedSearch(updatedFilters);
  };

  const handleFilterReset = () => {
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
