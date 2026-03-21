"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Input, Modal, Spin, Typography } from "antd";
import { Search } from "lucide-react";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
} from "@/components/networking";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import debounce from "lodash/debounce";

interface CommandPaletteProps {
  onSelectKey?: (key: KeyResponse) => void;
}

interface SearchResult {
  keys: KeyResponse[];
  total_count: number;
}

async function searchKeys(
  accessToken: string,
  query: string,
): Promise<SearchResult> {
  const baseUrl = getProxyBaseUrl();
  const params = new URLSearchParams({
    key_alias: query,
    page: "1",
    size: "10",
    return_full_object: "true",
    include_team_keys: "true",
    include_created_by_keys: "true",
    expand: "user",
  });
  const url = `${baseUrl ? `${baseUrl}/key/list` : "/key/list"}?${params}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(deriveErrorMessage(errorData));
  }
  return response.json();
}

const CommandPalette: React.FC<CommandPaletteProps> = ({ onSelectKey }) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KeyResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const { accessToken } = useAuthorized();
  const inputRef = useRef<any>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setResults([]);
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open]);

  const debouncedSearch = useCallback(
    debounce(async (searchQuery: string) => {
      if (!accessToken || !searchQuery.trim()) {
        setResults([]);
        setLoading(false);
        return;
      }
      try {
        const data = await searchKeys(accessToken, searchQuery);
        setResults(data.keys || []);
        setSelectedIndex(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300),
    [accessToken],
  );

  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
    };
  }, [debouncedSearch]);

  const handleQueryChange = (value: string) => {
    setQuery(value);
    if (value.trim()) {
      setLoading(true);
      debouncedSearch(value);
    } else {
      setResults([]);
      setLoading(false);
    }
  };

  const handleSelect = (key: KeyResponse) => {
    setOpen(false);
    onSelectKey?.(key);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && results[selectedIndex]) {
      e.preventDefault();
      handleSelect(results[selectedIndex]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  useEffect(() => {
    if (listRef.current) {
      const selected = listRef.current.querySelector(`[data-index="${selectedIndex}"]`);
      selected?.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  return (
    <Modal
      open={open}
      onCancel={() => setOpen(false)}
      footer={null}
      closable={false}
      width={600}
      styles={{ body: { padding: 0 } }}
      className="command-palette-modal"
    >
      <div onKeyDown={handleKeyDown}>
        <div className="p-3 border-b">
          <Input
            ref={inputRef}
            prefix={<Search size={16} className="text-gray-400" />}
            placeholder="Search keys by alias, ID, or user..."
            value={query}
            onChange={(e) => handleQueryChange(e.target.value)}
            variant="borderless"
            size="large"
            suffix={
              <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium text-gray-400 bg-gray-100 rounded border border-gray-200">
                ESC
              </kbd>
            }
          />
        </div>

        <div ref={listRef} className="max-h-[400px] overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Spin size="small" />
              <span className="ml-2 text-sm text-gray-500">Searching...</span>
            </div>
          )}

          {!loading && query && results.length === 0 && (
            <div className="py-8 text-center text-sm text-gray-500">
              No keys found for &ldquo;{query}&rdquo;
            </div>
          )}

          {!loading && !query && (
            <div className="py-8 text-center text-sm text-gray-400">
              Type to search for keys by alias, ID, or user
            </div>
          )}

          {!loading &&
            results.map((key, index) => (
              <div
                key={key.token}
                data-index={index}
                onClick={() => handleSelect(key)}
                className={`flex items-center gap-3 px-4 py-3 cursor-pointer border-b border-gray-50 transition-colors ${
                  index === selectedIndex ? "bg-blue-50" : "hover:bg-gray-50"
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm truncate">
                      {key.key_alias || "Unnamed Key"}
                    </span>
                    <span className="text-xs font-mono text-gray-400 truncate">
                      {key.key_name}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-0.5">
                    {key.team_id && (
                      <Typography.Text className="text-xs text-gray-400" ellipsis>
                        Team: {key.team_id}
                      </Typography.Text>
                    )}
                    {key.user_id && (
                      <Typography.Text className="text-xs text-gray-400" ellipsis>
                        User: {key.user_id}
                      </Typography.Text>
                    )}
                    <Typography.Text className="text-xs text-gray-400">
                      Spend: ${(key.spend ?? 0).toFixed(4)}
                    </Typography.Text>
                  </div>
                </div>
                <span className="text-xs text-gray-300 font-mono shrink-0">
                  {key.token?.slice(0, 8)}...
                </span>
              </div>
            ))}
        </div>

        {results.length > 0 && (
          <div className="px-4 py-2 border-t bg-gray-50 flex items-center gap-4 text-xs text-gray-400">
            <span>
              <kbd className="px-1 py-0.5 bg-white rounded border text-[10px]">↑↓</kbd> navigate
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-white rounded border text-[10px]">↵</kbd> select
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-white rounded border text-[10px]">esc</kbd> close
            </span>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default CommandPalette;
