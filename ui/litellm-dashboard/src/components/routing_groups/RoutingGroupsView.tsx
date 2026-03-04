"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button, Skeleton, Select } from "antd";
import { PlusOutlined } from "@ant-design/icons";
import RoutingGroupsTable from "./RoutingGroupsTable";
import RoutingGroupBuilder from "./RoutingGroupBuilder";
import { routingGroupListCall, routingGroupDeleteCall } from "@/components/networking";

interface RoutingGroupRow {
  routing_group_id: string;
  routing_group_name: string;
  routing_strategy: string;
  deployments: unknown[];
  is_active: boolean;
  created_at: string;
  description?: string;
  [key: string]: unknown;
}

interface RoutingGroupsViewProps {
  accessToken: string | null;
  userRole: string;
  userId: string;
}

const STRATEGY_OPTIONS = [
  { value: "all", label: "All Strategies" },
  { value: "priority-failover", label: "Priority Failover" },
  { value: "weighted", label: "Weighted" },
  { value: "cost-based-routing", label: "Cost-Based" },
  { value: "latency-based-routing", label: "Latency-Based" },
  { value: "least-busy", label: "Least Busy" },
  { value: "usage-based-routing-v2", label: "Usage-Based" },
  { value: "simple-shuffle", label: "Round Robin" },
];

export default function RoutingGroupsView({
  accessToken,
  userRole,
  userId,
}: RoutingGroupsViewProps) {
  const [showBuilder, setShowBuilder] = useState(false);
  const [editTarget, setEditTarget] = useState<Record<string, unknown> | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [groups, setGroups] = useState<RoutingGroupRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [nameSearch, setNameSearch] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 50;

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => {
      setDebouncedSearch(nameSearch);
    }, 200);
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [nameSearch]);

  useEffect(() => {
    if (!accessToken) return;
    setLoading(true);
    routingGroupListCall(accessToken, 1, 500)
      .then((resp) => {
        setGroups((resp.routing_groups as RoutingGroupRow[]) || []);
      })
      .catch(() => {
        setGroups([]);
      })
      .finally(() => setLoading(false));
  }, [accessToken, refreshKey]);

  const filteredData = useMemo(() => {
    return groups.filter((g) => {
      const nameMatch =
        !debouncedSearch ||
        g.routing_group_name.toLowerCase().includes(debouncedSearch.toLowerCase());
      const strategyMatch =
        selectedStrategy === "all" || g.routing_strategy === selectedStrategy;
      return nameMatch && strategyMatch;
    });
  }, [groups, debouncedSearch, selectedStrategy]);

  const totalCount = filteredData.length;
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
  const paginatedData = filteredData.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  useEffect(() => {
    setCurrentPage(1);
  }, [selectedStrategy, debouncedSearch]);

  const resetFilters = () => {
    setNameSearch("");
    setDebouncedSearch("");
    setSelectedStrategy("all");
    setCurrentPage(1);
  };

  const handleDelete = async (groupId: string) => {
    if (!accessToken) return;
    try {
      await routingGroupDeleteCall(accessToken, groupId);
      setGroups((prev) => prev.filter((g) => g.routing_group_id !== groupId));
    } catch (err) {
      console.error("Failed to delete routing group:", err);
    }
  };

  const handleEdit = (group: Record<string, unknown>) => {
    setEditTarget(group);
    setShowBuilder(true);
  };

  const handleClose = () => {
    setShowBuilder(false);
    setEditTarget(null);
  };

  const handleSuccess = () => {
    setShowBuilder(false);
    setEditTarget(null);
    setRefreshKey((k) => k + 1);
  };

  if (showBuilder) {
    return (
      <div className="w-full">
        <RoutingGroupBuilder
          accessToken={accessToken}
          editTarget={editTarget}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </div>
    );
  }

  return (
    <div className="w-full max-w-screen p-6 overflow-x-hidden box-border">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Routing Groups</h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setShowBuilder(true)}
        >
          Create Routing Group
        </Button>
      </div>

      <div className="bg-white rounded-lg shadow w-full max-w-full box-border">
        {/* Search and Filter Controls */}
        <div className="border-b px-6 py-4">
          <div className="flex flex-col space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="relative w-64">
                  <input
                    type="text"
                    placeholder="Search routing groups..."
                    className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    value={nameSearch}
                    onChange={(e) => setNameSearch(e.target.value)}
                  />
                  <svg
                    className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>

                <button
                  className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? "bg-gray-100" : ""}`}
                  onClick={() => setShowFilters(!showFilters)}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                    />
                  </svg>
                  Filters
                </button>

                <button
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  onClick={resetFilters}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  Reset Filters
                </button>
              </div>
            </div>

            {showFilters && (
              <div className="flex flex-wrap items-center gap-3 mt-3">
                <div className="w-64">
                  <Select
                    className="w-full"
                    value={selectedStrategy}
                    onChange={(value) => setSelectedStrategy(value)}
                    placeholder="Filter by Strategy"
                    showSearch
                    options={STRATEGY_OPTIONS}
                  />
                </div>
              </div>
            )}

            <div className="flex justify-between items-center">
              {loading ? (
                <Skeleton.Input active style={{ width: 184, height: 20 }} />
              ) : (
                <span className="text-sm text-gray-700">
                  {totalCount > 0
                    ? `Showing ${(currentPage - 1) * pageSize + 1} - ${Math.min(currentPage * pageSize, totalCount)} of ${totalCount} results`
                    : "Showing 0 results"}
                </span>
              )}

              <div className="flex items-center space-x-2">
                {loading ? (
                  <Skeleton.Button active style={{ width: 84, height: 30 }} />
                ) : (
                  <button
                    onClick={() => setCurrentPage((p) => p - 1)}
                    disabled={currentPage === 1}
                    className={`px-3 py-1 text-sm border rounded-md ${
                      currentPage === 1
                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    Previous
                  </button>
                )}

                {loading ? (
                  <Skeleton.Button active style={{ width: 56, height: 30 }} />
                ) : (
                  <button
                    onClick={() => setCurrentPage((p) => p + 1)}
                    disabled={currentPage >= totalPages}
                    className={`px-3 py-1 text-sm border rounded-md ${
                      currentPage >= totalPages
                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    Next
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        <RoutingGroupsTable
          data={paginatedData}
          loading={loading}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      </div>
    </div>
  );
}
