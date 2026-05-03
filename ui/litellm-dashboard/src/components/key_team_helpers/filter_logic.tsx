import { useEffect, useState } from "react";
import { Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import { fetchAllOrganizations, fetchAllTeams } from "./filter_helpers";
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

const DEFAULT_FILTERS: FilterState = {
  "Team ID": "",
  "Organization ID": "",
  "Key Alias": "",
  "User ID": "",
  "Sort By": "created_at",
  "Sort Order": "desc",
};

export function useFilterLogic({
  teams,
  organizations,
}: {
  teams: Team[] | null;
  organizations: Organization[] | null;
}) {
  const { accessToken } = useAuthorized();
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [allTeams, setAllTeams] = useState<Team[]>(teams || []);
  const [allOrganizations, setAllOrganizations] = useState<Organization[]>(organizations || []);

  useEffect(() => {
    if (!accessToken) return;

    const loadAllFilterData = async () => {
      const teamsData = await fetchAllTeams(accessToken);
      if (teamsData.length > 0) {
        setAllTeams(teamsData);
      }

      const orgsData = await fetchAllOrganizations(accessToken);
      if (orgsData.length > 0) {
        setAllOrganizations(orgsData);
      }
    };

    loadAllFilterData();
  }, [accessToken]);

  useEffect(() => {
    if (teams && teams.length > 0) {
      setAllTeams((prevTeams) => (prevTeams.length < teams.length ? teams : prevTeams));
    }
  }, [teams]);

  useEffect(() => {
    if (organizations && organizations.length > 0) {
      setAllOrganizations((prevOrgs) => (prevOrgs.length < organizations.length ? organizations : prevOrgs));
    }
  }, [organizations]);

  const handleFilterChange = (newFilters: Record<string, string>) => {
    setFilters({
      "Team ID": newFilters["Team ID"] || "",
      "Organization ID": newFilters["Organization ID"] || "",
      "Key Alias": newFilters["Key Alias"] || "",
      "User ID": newFilters["User ID"] || "",
      "Sort By": newFilters["Sort By"] || "created_at",
      "Sort Order": newFilters["Sort Order"] || "desc",
    });
  };

  const handleFilterReset = () => {
    setFilters(DEFAULT_FILTERS);
  };

  return {
    filters,
    allTeams,
    allOrganizations,
    handleFilterChange,
    handleFilterReset,
  };
}
