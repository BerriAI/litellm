/* @vitest-environment jsdom */
import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import KeyModelList from "./KeyModelList";
import * as useGetKeyModelsHook from "@/hooks/keys/useGetKeyModels";

vi.mock("@/hooks/keys/useGetKeyModels", () => ({
  useGetKeyModels: vi.fn(),
}));

const mockUseGetKeyModels = vi.mocked(useGetKeyModelsHook.useGetKeyModels);

describe("KeyModelList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render grouped models by default without requiring search", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "",
      setSearchInput: vi.fn(),
      defaultModelsQuery: {
        data: {
          model_display_sections: [
            { title: "All proxy models", section_kind: "all_proxy_models", models: ["m1", "m2"] },
            { title: "grp-a", section_kind: "access_group", models: ["m1"] },
          ],
          source: "all-proxy-models",
          resolved_config_entry_count: 2,
          matched_count: 2,
          models_truncated: false,
          all_team_models_without_team: false,
        },
        isLoading: false,
        isError: false,
        isSuccess: true,
      } as any,
      searchQuery: { data: undefined, isLoading: false, isFetching: false, isFetched: false } as any,
      hasActiveSearch: false,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("All proxy models")).toBeInTheDocument();
    expect(screen.getByText("access_group")).toBeInTheDocument();
    expect(screen.getByText("grp-a")).toBeInTheDocument();
    expect(screen.getAllByText("m1").length).toBeGreaterThanOrEqual(1);
    const innerCards = document.querySelectorAll(".ant-card-type-inner");
    expect(innerCards.length).toBeGreaterThanOrEqual(2);
  });

  it("should show explanation icon next to All team models when no team is assigned", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "",
      setSearchInput: vi.fn(),
      defaultModelsQuery: {
        data: {
          model_display_sections: [
            { title: "All team models", section_kind: "all_team_models", models: ["a"] },
          ],
          source: "all-team-models",
          resolved_config_entry_count: 1,
          matched_count: 1,
          models_truncated: false,
          all_team_models_without_team: true,
        },
        isLoading: false,
        isError: false,
        isSuccess: true,
      } as any,
      searchQuery: { data: undefined, isLoading: false, isFetching: false, isFetched: false } as any,
      hasActiveSearch: false,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    expect(screen.getByLabelText("Why this matters")).toBeInTheDocument();
    expect(screen.getByText("All team models")).toBeInTheDocument();
  });

  it("should use a scroll-capped list region with overflow-y-auto", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "",
      setSearchInput: vi.fn(),
      defaultModelsQuery: {
        data: {
          model_display_sections: [
            { title: "Other models", section_kind: "ungrouped", models: ["x"] },
          ],
          source: "no-default-models",
          resolved_config_entry_count: 1,
          matched_count: 1,
          models_truncated: false,
          all_team_models_without_team: false,
        },
        isLoading: false,
        isError: false,
        isSuccess: true,
      } as any,
      searchQuery: { data: undefined, isLoading: false, isFetching: false, isFetched: false } as any,
      hasActiveSearch: false,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    const scroll = screen.getByTestId("key-model-list-scroll");
    expect(scroll.className).toMatch(/overflow-y-auto/);
    expect(scroll.className).toMatch(/min-h-0/);
  });

  it("should show search results when hasActiveSearch", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "gpt",
      setSearchInput: vi.fn(),
      defaultModelsQuery: { data: undefined, isLoading: false, isError: false, isSuccess: true } as any,
      searchQuery: {
        data: {
          model_display_sections: [
            { title: "Other models", section_kind: "ungrouped", models: ["gpt-4"] },
          ],
          source: "no-default-models",
          resolved_config_entry_count: 1,
          matched_count: 1,
          models_truncated: false,
          all_team_models_without_team: false,
        },
        isLoading: false,
        isFetching: false,
        isFetched: true,
        isSuccess: true,
      } as any,
      hasActiveSearch: true,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
  });

  it("should show nothing found when search succeeds with zero matches", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "zzz",
      setSearchInput: vi.fn(),
      defaultModelsQuery: { data: undefined, isLoading: false, isError: false, isSuccess: true } as any,
      searchQuery: {
        data: {
          model_display_sections: [],
          source: "no-default-models",
          resolved_config_entry_count: 5,
          matched_count: 0,
          models_truncated: false,
          all_team_models_without_team: false,
        },
        isLoading: false,
        isFetching: false,
        isFetched: true,
        isSuccess: true,
      } as any,
      hasActiveSearch: true,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    expect(screen.getByText(/Nothing found for "zzz"/i)).toBeInTheDocument();
  });

  it("should render all proxy scope and access group sections together", () => {
    mockUseGetKeyModels.mockReturnValue({
      searchInput: "m",
      setSearchInput: vi.fn(),
      defaultModelsQuery: { data: undefined, isLoading: false, isError: false, isSuccess: true } as any,
      searchQuery: {
        data: {
          model_display_sections: [
            { title: "All proxy models", section_kind: "all_proxy_models", models: ["m1", "m2"] },
            { title: "grp-a", section_kind: "access_group", models: ["m1"] },
            { title: "grp-b", section_kind: "access_group", models: ["m2", "m1"] },
          ],
          source: "all-proxy-models",
          resolved_config_entry_count: 2,
          matched_count: 2,
          models_truncated: false,
          all_team_models_without_team: false,
        },
        isLoading: false,
        isFetching: false,
        isFetched: true,
        isSuccess: true,
      } as any,
      hasActiveSearch: true,
      isInitialLoading: false,
      searchInputLoading: false,
    });

    render(<KeyModelList key_id="k1" />);
    expect(screen.getByText("All proxy models")).toBeInTheDocument();
    expect(screen.getAllByText("access_group").length).toBeGreaterThanOrEqual(2);
    const m1Tags = screen.getAllByText("m1");
    expect(m1Tags.length).toBeGreaterThanOrEqual(2);
  });
});
