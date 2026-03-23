import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { Select, Space, Typography } from "antd";
import { useMemo, useState, type UIEvent } from "react";

const { Text } = Typography;

export interface PaginatedModelSelectProps {
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

export const PaginatedModelSelect = ({
  value,
  onChange,
  placeholder = "Select a model",
  style,
  pageSize = 50,
  allowClear = true,
  disabled = false,
}: PaginatedModelSelectProps) => {
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
  } = useInfiniteModelInfo(pageSize, debouncedSearch || undefined);

  const options = useMemo(() => {
    if (!data?.pages) return [];

    const seen = new Set<string>();
    const result: { label: string; value: string; modelName: string; modelId: string }[] = [];

    for (const page of data.pages) {
      for (const model of page.data) {
        const modelId = model.model_info?.id ?? "";
        const modelName = model.model_name ?? "";

        // Dedupe by id - skip models without id (can't uniquely identify)
        if (!modelId || seen.has(modelId)) continue;
        seen.add(modelId);

        result.push({
          label: modelName ? `${modelName} (${modelId})` : modelId,
          value: modelId,
          modelName,
          modelId,
        });
      }
    }

    return result;
  }, [data]);

  const optionRender = (option: { data: { modelName: string; modelId: string; label: string } }) => {
    const { modelName, modelId } = option.data;

    return (
      <>
        {modelName ? (
          <Space direction="vertical">
            <Space direction="horizontal">
              <Text strong>Model name:</Text>
              <Text ellipsis>{modelName}</Text>
            </Space>
            <Text ellipsis type="secondary" >
              Model ID: {modelId}
            </Text>
          </Space>
        ) : (
          <Text ellipsis type="secondary">Model ID: {modelId}</Text>
        )}
      </>
    );
  };

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

  const handleChange = (v: string | string[] | null) => {
    const normalized =
      typeof v === "string" ? v : Array.isArray(v) ? v[0] ?? "" : "";
    onChange?.(normalized);
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
      notFoundContent={isLoading ? <LoadingOutlined spin /> : "No models found"}
      options={options}
      optionRender={optionRender}
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
