import { useState, useEffect } from 'react';
import { keyListCall, Organization } from '../networking';
import { KeyResponse, Team } from './key_list';

interface UseKeyAliasListProps {
  selectedTeam?: Team;
  currentOrg: Organization | null;
  accessToken: string;
}

interface UseKeyAliasListReturn {
  keyAliases: { alias: string, token: string }[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

/**
 * Custom hook to fetch and manage a list of unique key aliases across all pages
 */
const useKeyAliasList = ({
  selectedTeam,
  currentOrg,
  accessToken,
}: UseKeyAliasListProps): UseKeyAliasListReturn => {
  const [keyAliases, setKeyAliases] = useState<{ alias: string, token: string }[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchAllKeyAliases = async (): Promise<void> => {
    try {
      if (!accessToken) {
        console.log("No access token available");
        return;
      }
      
      setIsLoading(true);
      
      // Track all unique key aliases
      const uniqueKeyAliases = new Map<string, string>();
      let currentPage = 1;
      let hasMorePages = true;
      
      // Fetch all pages to get complete list of keys
      while (hasMorePages) {
        const data = await keyListCall(
          accessToken,
          currentOrg?.organization_id || null,
          selectedTeam?.team_id || "",
          currentPage,
          100 // Use larger page size to reduce number of requests
        );
        
        // Add unique key aliases to our map
        data.keys.forEach((key: KeyResponse) => {
          if (key.key_alias && !uniqueKeyAliases.has(key.key_alias)) {
            uniqueKeyAliases.set(key.key_alias, key.token);
          }
        });
        
        // Check if we've reached the last page
        if (currentPage >= data.total_pages) {
          hasMorePages = false;
        } else {
          currentPage++;
        }
      }
      
      // Convert map to array of objects
      const aliasArray = Array.from(uniqueKeyAliases.entries()).map(([alias, token]) => ({
        alias,
        token
      }));
      
      // Sort alphabetically by alias
      aliasArray.sort((a, b) => a.alias.localeCompare(b.alias));
      
      setKeyAliases(aliasArray);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("An error occurred fetching key aliases"));
      console.error("Error fetching key aliases:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAllKeyAliases();
  }, [selectedTeam, currentOrg, accessToken]);

  return {
    keyAliases,
    isLoading,
    error,
    refresh: fetchAllKeyAliases
  };
};

export default useKeyAliasList; 