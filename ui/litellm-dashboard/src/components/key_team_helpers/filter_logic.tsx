import { useEffect, useState } from "react";
import { KeyResponse } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import { Team } from "../key_team_helpers/key_list";
import { useQuery } from "@tanstack/react-query";
import { fetchAllKeyAliases, fetchAllOrganizations, fetchAllTeams } from "./filter_helpers";
import { Setter } from "@/types";


export interface FilterState {
  'Team ID': string;
  'Organization ID': string;
  'Key Alias': string;
  [key: string]: string;
}

export function useFilterLogic({
  keys,
  teams,
  organizations,
  accessToken,
  setSelectedTeam,
  setCurrentOrg,
  setSelectedKeyAlias
}: {
  keys: KeyResponse[];
  teams: Team[] | null;
  organizations: Organization[] | null;
  accessToken: string | null;
  setSelectedTeam: (team: Team | null) => void;
  setCurrentOrg: React.Dispatch<React.SetStateAction<Organization | null>>;
  setSelectedKeyAlias: Setter<string | null>
}) {
  const [filters, setFilters] = useState<FilterState>({
    'Team ID': '',
    'Organization ID': '',
    'Key Alias': ''
  });
  const [allTeams, setAllTeams] = useState<Team[]>(teams || []);
  const [allOrganizations, setAllOrganizations] = useState<Organization[]>(organizations || []);
  const [filteredKeys, setFilteredKeys] = useState<KeyResponse[]>(keys);

  // Apply filters to keys whenever keys or filters change
  useEffect(() => {
    if (!keys) {
      setFilteredKeys([]);
      return;
    }

    let result = [...keys];

    // Apply Team ID filter
    if (filters['Team ID']) {
      result = result.filter(key => key.team_id === filters['Team ID']);
    }

    // Apply Organization ID filter
    if (filters['Organization ID']) {
      result = result.filter(key => key.organization_id === filters['Organization ID']);
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
    queryKey: ['allKeys'],
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');
      return await fetchAllKeyAliases(accessToken);
    },
    enabled: !!accessToken
  });
  const allKeyAliases = queryAllKeysQuery.data || []

  // Update teams and organizations when props change
  useEffect(() => {
    if (teams && teams.length > 0) {
      setAllTeams(prevTeams => {
        // Only update if we don't already have a larger set of teams
        return prevTeams.length < teams.length ? teams : prevTeams;
      });
    }
  }, [teams]);

  useEffect(() => {
    if (organizations && organizations.length > 0) {
      setAllOrganizations(prevOrgs => {
        // Only update if we don't already have a larger set of organizations
        return prevOrgs.length < organizations.length ? organizations : prevOrgs;
      });
    }
  }, [organizations]);

  const handleFilterChange = (newFilters: Record<string, string>) => {
    // Update filters state
    setFilters({
      'Team ID': newFilters['Team ID'] || '',
      'Organization ID': newFilters['Organization ID'] || '',
      'Key Alias': newFilters['Key Alias'] || ''
    });
  
    // Handle Team change
    if (newFilters['Team ID']) {
      const selectedTeamData = allTeams?.find(team => team.team_id === newFilters['Team ID']);
      if (selectedTeamData) {
        setSelectedTeam(selectedTeamData);
      }
    }
  
    // Handle Org change
    if (newFilters['Organization ID']) {
      const selectedOrg = allOrganizations?.find(org => org.organization_id === newFilters['Organization ID']);
      if (selectedOrg) {
        setCurrentOrg(selectedOrg);
      }
    }

    const keyAlias = newFilters['Key Alias'];
    const selectedKeyAlias = keyAlias
      ? allKeyAliases.find((k) => k === keyAlias) || null
      : null;
    setSelectedKeyAlias(selectedKeyAlias)
  };

  const handleFilterReset = () => {
    // Reset filters state
    setFilters({
      'Team ID': '',
      'Organization ID': '',
      'Key Alias': ''
    });
    
    // Reset selections
    setSelectedTeam(null);
    setCurrentOrg(null);
    setSelectedKeyAlias(null);
  };

  return {
    filters,
    filteredKeys,
    allKeyAliases,
    allTeams,
    allOrganizations,
    handleFilterChange,
    handleFilterReset
  };
}
