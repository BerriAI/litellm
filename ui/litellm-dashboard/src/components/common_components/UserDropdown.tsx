import React, { useMemo, useState } from "react";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { useInfiniteUsers, useUserLookup } from "@/app/(dashboard)/hooks/users/useUsers";
import type { UserInfo } from "@/components/networking";

interface UserDropdownProps {
  value?: string | null;
  onChange: (userId: string | null) => void;
  disabled?: boolean;
  pageSize?: number;
  id?: string;
  searchField?: "email" | "alias";
  pinnedUser?: Pick<UserInfo, "user_id" | "user_alias" | "user_email"> | null;
}

export const userOptionLabel = (user: Pick<UserInfo, "user_id" | "user_alias" | "user_email">): string => {
  if (user.user_alias) return `${user.user_alias} (${user.user_id})`;
  if (user.user_email) return `${user.user_email} (${user.user_id})`;
  return user.user_id;
};

const UserDropdown: React.FC<UserDropdownProps> = ({
  value,
  onChange,
  disabled,
  pageSize = 50,
  id,
  searchField = "email",
  pinnedUser,
}) => {
  const [search, setSearch] = useState("");

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteUsers(
    pageSize,
    searchField === "email" ? search || undefined : undefined,
    searchField === "alias" ? search || undefined : undefined,
  );

  const loadedOptions = useMemo<SearchSelectOption[]>(() => {
    const byId = new Map<string, SearchSelectOption>();
    for (const user of (data?.pages ?? []).flatMap((page) => page.users)) {
      if (user.user_id === pinnedUser?.user_id) continue;
      if (byId.has(user.user_id)) continue;
      byId.set(user.user_id, { value: user.user_id, label: userOptionLabel(user) });
    }
    return Array.from(byId.values());
  }, [data, pinnedUser?.user_id]);

  const selectedIsLoaded = loadedOptions.some((option) => option.value === value);
  const selectedIsPinned = pinnedUser != null && value === pinnedUser.user_id;
  const { data: selectedUser } = useUserLookup(value && !selectedIsLoaded && !selectedIsPinned ? value : null);

  const options = useMemo<SearchSelectOption[]>(() => {
    const selectedUserIsNeeded = value != null && !selectedIsLoaded && !selectedIsPinned;
    const includeSelectedUser = selectedUserIsNeeded && selectedUser != null;
    const selectedOption = includeSelectedUser
      ? [{ value: selectedUser.user_id, label: userOptionLabel(selectedUser) }]
      : [];
    return [
      ...(pinnedUser ? [{ value: pinnedUser.user_id, label: `${userOptionLabel(pinnedUser)} (you)` }] : []),
      ...selectedOption,
      ...loadedOptions,
    ];
  }, [pinnedUser, value, selectedIsLoaded, selectedIsPinned, selectedUser, loadedOptions]);

  return (
    <div data-testid="user-dropdown">
      <PaginatedSearchSelect
        options={options}
        value={value}
        onValueChange={onChange}
        onSearchChange={setSearch}
        onLoadMore={fetchNextPage}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder={`Search users by ${searchField}…`}
        emptyText="No users found"
        loadingText="Loading users…"
        disabled={disabled}
        inputId={id}
      />
    </div>
  );
};

export default UserDropdown;
