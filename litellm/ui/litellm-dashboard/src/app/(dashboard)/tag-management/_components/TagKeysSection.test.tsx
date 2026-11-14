import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { KeyResponse } from "@/components/key_team_helpers/key_list";

import TagKeysSection, { TAG_KEYS_PAGE_SIZE } from "./TagKeysSection";

const mockUseKeys = vi.fn();
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: (...args: unknown[]) => mockUseKeys(...args),
}));

const makeKey = (overrides: Partial<KeyResponse>): KeyResponse =>
  ({ token: "tok", key_alias: "", key_name: "sk-...0000", team_id: null, spend: 0, ...overrides }) as KeyResponse;

const loaded = (keys: KeyResponse[], totalCount = keys.length) => ({
  data: { keys, total_count: totalCount, current_page: 1, total_pages: 1 },
  isLoading: false,
  isError: false,
});

describe("TagKeysSection", () => {
  beforeEach(() => {
    mockUseKeys.mockReset();
  });

  it("should request the first page of keys filtered by the tag name", () => {
    mockUseKeys.mockReturnValue(loaded([]));
    render(<TagKeysSection tagName="prod-batch" />);
    expect(mockUseKeys).toHaveBeenCalledWith(1, TAG_KEYS_PAGE_SIZE, { tag: "prod-batch" });
  });

  it("should list each key with its alias linking to the key's page, its team and its spend", () => {
    const teamKey: Partial<KeyResponse> = { token: "tok-1", key_alias: "batch-key-1", team_id: "team-a", spend: 1.5 };
    mockUseKeys.mockReturnValue(loaded([makeKey(teamKey), makeKey({ token: "tok-2", key_alias: "batch-key-2" })]));
    render(<TagKeysSection tagName="prod-batch" />);

    expect(screen.getByRole("link", { name: "batch-key-1" })).toHaveAttribute("href", "/ui/api-keys?key=tok-1");
    expect(screen.getByRole("link", { name: "batch-key-2" })).toHaveAttribute("href", "/ui/api-keys?key=tok-2");
    const firstRow = screen.getByRole("row", { name: /batch-key-1/ });
    expect(within(firstRow).getByText("team-a")).toBeInTheDocument();
    expect(within(firstRow).getByText("1.5000")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /batch-key-2/ })).getByText("-")).toBeInTheDocument();
  });

  it("should fall back to the masked key name when a key has no alias", () => {
    mockUseKeys.mockReturnValue(loaded([makeKey({ token: "tok-3", key_alias: "", key_name: "sk-...wxyz" })]));
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.getByRole("link", { name: "sk-...wxyz" })).toHaveAttribute("href", "/ui/api-keys?key=tok-3");
  });

  it("should say no virtual keys use the tag when none carry it", () => {
    mockUseKeys.mockReturnValue(loaded([]));
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.getByText("No virtual keys use this tag")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("should show a loading message while the keys are loading", () => {
    mockUseKeys.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.getByText("Loading virtual keys...")).toBeInTheDocument();
    expect(screen.queryByText("No virtual keys use this tag")).not.toBeInTheDocument();
  });

  it("should show an error message instead of the empty message when the keys fail to load", () => {
    mockUseKeys.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.getByText("Could not load the virtual keys for this tag")).toBeInTheDocument();
    expect(screen.queryByText("No virtual keys use this tag")).not.toBeInTheDocument();
  });

  it("should say how many keys are shown when the tag has more keys than one page", () => {
    mockUseKeys.mockReturnValue(loaded([makeKey({ token: "tok-1", key_alias: "batch-key-1" })], 250));
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.getByText("Showing the 1 most recently created of 250 keys")).toBeInTheDocument();
  });

  it("should not show a count note when every key fits on the page", () => {
    mockUseKeys.mockReturnValue(loaded([makeKey({ token: "tok-1", key_alias: "batch-key-1" })]));
    render(<TagKeysSection tagName="prod-batch" />);
    expect(screen.queryByText(/most recently created/)).not.toBeInTheDocument();
  });
});
