import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { Select } from "antd";
import { useMemo, useState, type UIEvent } from "react";

export interface PaginatedKeyAliasSelectProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  style?: React.CSSProperties;
  pageSize?: number;
  allowClear?: boolean;
  disabled?: boolean;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

export const PaginatedKeyAliasSelect = ({
  value,
  onChange,
  placeholder = "Select a key alias",
  style,
  pageSize = 50,
  allowClear = true,
  disabled = false,
}: PaginatedKeyAliasSelectProps) => {
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
  } = useInfiniteKeyAliases(pageSize, debouncedSearch || undefined);

  const options = useMemo(() => {
    if (!data?.pages) return [];

    const seen = new Set<string>();
    const result: { label: string; value: string }[] = [];

    for (const page of data.pages) {
      for (const alias of page.aliases) {
        if (!alias || seen.has(alias)) continue;
        seen.add(alias);
        result.push({ label: alias, value: alias });
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

  const handleSearch = (value: string) => {
    setSearchInput(value);
    setDebouncedSearch(value);
  };

  const handleChange = (v: string | null) => {
    onChange?.(v ?? "");
  };

  return (
    <Select
      value={value || undefined}
      onChange={handleChange}
      placeholder={placeholder}
      style={{ width: "100%", ...style }}
      allowClear={allowClear}
      disabled={disabled}
      showSearch
      filterOption={false}
      onSearch={handleSearch}
      searchValue={searchInput}
      onPopupScroll={handlePopupScroll}
      loading={isLoading}
      notFoundContent={isLoading ? <LoadingOutlined spin /> : "No key aliases found"}
      options={options}
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
    />
  );
};
