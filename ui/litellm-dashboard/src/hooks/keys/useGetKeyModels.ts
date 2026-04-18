import { useQuery } from '@tanstack/react-query';
import { useDebouncedState } from '@tanstack/react-pacer/debouncer';
import { useCallback, useState } from 'react';
import { fetchKeyModelCall } from '@/components/networking';
import useAuthorized from '@/app/(dashboard)/hooks/useAuthorized';

/** Wait after last keystroke before calling the search API (avoids spzzy refetches). */
const SEARCH_DEBOUNCE_MS = 450;

export const useGetKeyModels = (key_id: string) => {
  const { accessToken } = useAuthorized();
  const [searchInput, setSearchInputState] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState('', {
    wait: SEARCH_DEBOUNCE_MS,
  });

  const setSearchInput = useCallback(
    (value: string) => {
      setSearchInputState(value);
      setDebouncedSearch(value);
    },
    [setDebouncedSearch],
  );

  const trimmedDebounced = debouncedSearch.trim();
  const hasActiveSearch = trimmedDebounced.length > 0;
  /** True while the user is still typing and the debounced value has not caught up yet. */
  const isSearchDebouncing = searchInput.trim() !== trimmedDebounced;

  const defaultModelsQuery = useQuery({
    queryKey: ['keyModelsDefault', key_id],
    queryFn: () => {
      if (!accessToken) throw new Error('Access Token required');
      return fetchKeyModelCall(accessToken, key_id);
    },
    enabled: Boolean(accessToken && key_id),
  });

  const searchQuery = useQuery({
    queryKey: ['keyModelsSearch', key_id, trimmedDebounced],
    queryFn: () => {
      if (!accessToken) throw new Error('Access Token required');
      return fetchKeyModelCall(accessToken, key_id, { search: trimmedDebounced });
    },
    enabled: Boolean(accessToken && key_id && hasActiveSearch),
  });

  const isSearchFetching = hasActiveSearch && searchQuery.isFetching;
  const searchInputLoading = isSearchDebouncing || isSearchFetching;

  return {
    searchInput,
    setSearchInput,
    debouncedSearch: trimmedDebounced,
    defaultModelsQuery,
    searchQuery,
    hasActiveSearch,
    isInitialLoading: defaultModelsQuery.isLoading,
    searchInputLoading,
  };
};
