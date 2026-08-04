"use client";

import { useRef, useState } from "react";

import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { deriveErrorMessage, directoryUsersSearchCall } from "../networking";

const MIN_QUERY_LENGTH = 2;
const INPUT_ID = "ad-directory-user-search";

interface AdDirectoryUserSearchProps {
  accessToken: string;
  value?: string;
  onChange?: (email: string) => void;
  id?: string;
}

export function AdDirectoryUserSearch({ accessToken, value, onChange, id }: AdDirectoryUserSearchProps) {
  const [options, setOptions] = useState<SearchSelectOption[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Discards out-of-order search responses.
  const requestIdRef = useRef(0);

  const handleSearchChange = async (query: string) => {
    const requestId = ++requestIdRef.current;

    if (query.trim().length < MIN_QUERY_LENGTH) {
      setOptions([]);
      setError(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const users = await directoryUsersSearchCall(accessToken, query.trim());
      if (requestId !== requestIdRef.current) {
        return;
      }
      setOptions(
        users.map((user) => ({
          label: user.display_name || user.email,
          value: user.email,
          sublabel: user.display_name ? user.email : undefined,
        })),
      );
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      console.error("Error searching directory users:", err);
      setOptions([]);
      setError(deriveErrorMessage(err));
    } finally {
      if (requestId === requestIdRef.current) {
        setIsLoading(false);
      }
    }
  };

  return (
    <PaginatedSearchSelect
      inputId={id ?? INPUT_ID}
      options={options}
      value={value}
      onValueChange={(email) => onChange?.(email)}
      onSearchChange={handleSearchChange}
      onLoadMore={() => {}}
      isLoading={isLoading}
      placeholder="Search by name or email"
      loadingText="Searching..."
      emptyText={error ? `Directory search failed: ${error}` : "No users found"}
    />
  );
}
