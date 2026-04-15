import React, { useMemo, useState, type UIEvent } from "react";
import { Select, Typography } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";

const { Text } = Typography;

interface UserSingleSelectProps {
  value?: string | null;
  onChange?: (value: string | null) => void;
  disabled?: boolean;
  pageSize?: number;
  placeholder?: string;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

/**
 * A single-select dropdown for users with server-side debounced search and
 * infinite scroll. Mirrors TeamMultiSelect but uses single-select mode to
 * match the `/user/daily/activity` API contract that accepts one user_id.
 */
const UserSingleSelect: React.FC<UserSingleSelectProps> = ({
  value,
  onChange,
  disabled,
  pageSize = 50,
  placeholder = "Search users by email or ID...",
}) => {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_MS,
  });

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
  } = useInfiniteUsers(pageSize, debouncedSearch || undefined);

  const userOptions = useMemo(() => {
    if (!data?.pages) return [];
    const seen = new Set<string>();
    const result: { value: string; label: string; email: string | null }[] = [];
    for (const page of data.pages) {
      for (const user of page.users) {
        if (seen.has(user.user_id)) continue;
        seen.add(user.user_id);
        result.push({
          value: user.user_id,
          label: user.user_alias
            ? `${user.user_alias} (${user.user_id})`
            : user.user_email
              ? `${user.user_email} (${user.user_id})`
              : user.user_id,
          email: user.user_email ?? null,
        });
      }
    }
    return result;
  }, [data]);

  const handlePopupScroll = (e: UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio =
      (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (scrollRatio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  const handleSearch = (val: string) => {
    setSearchInput(val);
    setDebouncedSearch(val);
  };

  return (
    <Select
      showSearch
      allowClear
      placeholder={placeholder}
      value={value ?? undefined}
      onChange={(val: string | undefined) => onChange?.(val ?? null)}
      disabled={disabled}
      filterOption={false}
      onSearch={handleSearch}
      searchValue={searchInput}
      onPopupScroll={handlePopupScroll}
      loading={isLoading}
      notFoundContent={isLoading ? <LoadingOutlined spin /> : "No users found"}
      style={{ width: "100%" }}
      popupRender={(menu) => (
        <>
          {menu}
          {isFetchingNextPage && (
            <div style={{ textAlign: "center", padding: 8 }}>
              <LoadingOutlined spin />
            </div>
          )}
        </>
      )}
    >
      {userOptions.map((opt) => (
        <Select.Option key={opt.value} value={opt.value}>
          {opt.email ? (
            <>
              <span className="font-medium">{opt.email}</span>{" "}
              <Text type="secondary">({opt.value})</Text>
            </>
          ) : (
            <span>{opt.value}</span>
          )}
        </Select.Option>
      ))}
    </Select>
  );
};

export default UserSingleSelect;
