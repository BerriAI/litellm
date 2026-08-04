import React, { useEffect, useRef, useState } from "react";
import { AutoComplete, Form, Typography } from "antd";
import { deriveErrorMessage, directoryUsersSearchCall } from "../networking";
import type { DirectoryUser } from "../networking";

const { Text } = Typography;

const MIN_QUERY_LENGTH = 2;
const SEARCH_DEBOUNCE_MS = 300;

interface DirectoryUserOption {
  label: React.ReactNode;
  value: string;
  user: DirectoryUser;
}

interface AdDirectoryUserSearchProps {
  accessToken: string;
  onSelectUser: (user: DirectoryUser) => void;
  label?: string;
  name?: string;
  /** Bump this (e.g. on modal close) to clear search state and the
   * selected value - antd Select otherwise keeps the previously selected
   * option's rendered label cached internally. */
  resetSignal: number;
}

export const AdDirectoryUserSearch: React.FC<AdDirectoryUserSearchProps> = ({
  accessToken,
  onSelectUser,
  label = "User Email",
  name = "user_email",
  resetSignal,
}) => {
  const [options, setOptions] = useState<DirectoryUserOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedValue, setSelectedValue] = useState<string | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Discards out-of-order search responses.
  const requestIdRef = useRef(0);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    requestIdRef.current++;
    setOptions([]);
    setError(null);
    setLoading(false);
    setSelectedValue(null);
  }, [resetSignal]);

  const handleSearch = (query: string) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    const requestId = ++requestIdRef.current;

    if (query.trim().length < MIN_QUERY_LENGTH) {
      setOptions([]);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    timeoutRef.current = setTimeout(async () => {
      try {
        const users = await directoryUsersSearchCall(accessToken, query.trim());
        if (requestId !== requestIdRef.current) {
          return;
        }
        setOptions(
          users.map((user) => ({
            value: user.email,
            user,
            label: (
              <div className="flex flex-col">
                <Text>{user.display_name || user.email}</Text>
                {user.display_name && <Text type="secondary">{user.email}</Text>}
              </div>
            ),
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
          setLoading(false);
        }
      }
    }, SEARCH_DEBOUNCE_MS);
  };

  const handleSelect = (value: string, option: DirectoryUserOption) => {
    setSelectedValue(value);
    onSelectUser(option.user);
  };

  const notFoundContent = loading ? (
    "Searching..."
  ) : error ? (
    <Text type="danger">Directory search failed: {error}</Text>
  ) : (
    "No users found"
  );

  return (
    <Form.Item label={label} name={name}>
      <AutoComplete
        key={resetSignal}
        placeholder="Search by name or email"
        onSearch={handleSearch}
        onSelect={(value, option) => handleSelect(value, option as DirectoryUserOption)}
        onChange={(value) => setSelectedValue(value || null)}
        value={selectedValue}
        options={options}
        allowClear
        notFoundContent={notFoundContent}
      />
    </Form.Item>
  );
};
