import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../tests/test-utils";

import RoutingGroups from "./index";
import type { RoutingGroup } from "./types";
import { useRoutingGroups, useSaveRoutingGroups } from "@/app/(dashboard)/hooks/routingGroups/useRoutingGroups";
import { toast } from "@/lib/toast";

vi.mock("@/app/(dashboard)/hooks/routingGroups/useRoutingGroups", () => ({
  useRoutingGroups: vi.fn(),
  useSaveRoutingGroups: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/router/useRouterFields", () => ({
  useRouterFields: () => ({ data: undefined }),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelHub: () => ({ data: undefined }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/app/(dashboard)/hooks/proxySettings/useProxySettings", () => ({
  __esModule: true,
  default: () => ({ PROXY_BASE_URL: "https://proxy.example.com" }),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn(), fromError: vi.fn() },
}));

const prodGroup: RoutingGroup = {
  group_name: "prod-group",
  models: ["gpt-4o"],
  routing_strategy: "usage-based-routing",
};

const devGroup: RoutingGroup = {
  group_name: "dev-group",
  models: ["gpt-4o-mini"],
  routing_strategy: "simple-shuffle",
};

const setup = (
  overrides: { mutateAsync?: ReturnType<typeof vi.fn>; isPending?: boolean; groups?: RoutingGroup[] } = {},
) => {
  const mutateAsync = overrides.mutateAsync ?? vi.fn().mockResolvedValue(undefined);
  vi.mocked(useRoutingGroups).mockReturnValue({
    data: { routingGroups: overrides.groups ?? [prodGroup, devGroup], availableStrategies: [] },
    isLoading: false,
    refetch: vi.fn(),
    isFetching: false,
  } as unknown as ReturnType<typeof useRoutingGroups>);
  vi.mocked(useSaveRoutingGroups).mockReturnValue({
    mutateAsync,
    isPending: overrides.isPending ?? false,
  } as unknown as ReturnType<typeof useSaveRoutingGroups>);
  return { mutateAsync };
};

const USAGE_PANEL_TITLE = "How routing works for this group";

const renderRetainingMountWrites = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  render(<RoutingGroups />, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        {children}
      </NuqsTestingAdapter>
    ),
  });

const visibleGroupNames = () =>
  screen
    .getAllByRole("row")
    .map((row) => row.getAttribute("data-row-id"))
    .filter((id): id is string => id !== null);

const openDeleteConfirm = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByTestId("routing-group-actions-prod-group"));
  await user.click(await screen.findByTestId("routing-group-action-delete"));
  return screen.getByRole("dialog");
};

describe("RoutingGroups delete confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not show the confirmation until a group is chosen for deletion", () => {
    setup();
    renderWithProviders(<RoutingGroups />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Delete routing group?")).not.toBeInTheDocument();
  });

  it("should name the group being deleted in the confirmation", async () => {
    const user = userEvent.setup();
    setup();
    renderWithProviders(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);

    expect(within(dialog).getByText("Delete routing group?")).toBeInTheDocument();
    expect(within(dialog).getByText("prod-group")).toBeInTheDocument();
    expect(within(dialog).getByText(/This cannot be undone/)).toBeInTheDocument();
  });

  it("should save the remaining groups and report success when confirmed", async () => {
    const user = userEvent.setup();
    const { mutateAsync } = setup();
    renderWithProviders(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    expect(mutateAsync).toHaveBeenCalledWith([devGroup]);
    expect(toast.success).toHaveBeenCalledWith('Deleted routing group "prod-group"');
  });

  it("should report the failure and keep the confirmation open when the save rejects", async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockRejectedValue(new Error("boom"));
    setup({ mutateAsync });
    renderWithProviders(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    expect(toast.error).toHaveBeenCalledWith("boom");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("should dismiss without saving when cancelled", async () => {
    const user = userEvent.setup();
    const { mutateAsync } = setup();
    renderWithProviders(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(mutateAsync).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("RoutingGroups URL state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setup();
  });

  it("filters by group_search from the URL", () => {
    renderWithProviders(<RoutingGroups />, { searchParams: "?group_search=shuffle" });
    expect(screen.getByPlaceholderText("Search groups...")).toHaveValue("shuffle");
    expect(visibleGroupNames()).toEqual(["dev-group"]);
    expect(screen.getByText("Showing 1 result")).toBeInTheDocument();
  });

  it("writes the search to group_search and removes it when cleared", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<RoutingGroups />, { onUrlUpdate });

    fireEvent.change(screen.getByPlaceholderText("Search groups..."), { target: { value: "prod" } });
    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("group_search")).toBe("prod"));
    expect(visibleGroupNames()).toEqual(["prod-group"]);

    await user.click(screen.getByRole("button", { name: "Clear search" }));
    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("group_search")).toBe(false));
    expect(visibleGroupNames()).toEqual(["prod-group", "dev-group"]);
  });

  it("returns the table to the first page when the search changes", async () => {
    setup({
      groups: Array.from({ length: 30 }, (_, index) => ({
        group_name: `group-${String(index).padStart(2, "0")}`,
        models: ["gpt-4o"],
        routing_strategy: "simple-shuffle",
      })),
    });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<RoutingGroups />, { searchParams: "?page=2&sort_by=group_name", onUrlUpdate });
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");

    fireEvent.change(screen.getByPlaceholderText("Search groups..."), { target: { value: "group" } });
    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("group_search")).toBe("group"));
    const params = onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
    expect(params?.has("page")).toBe(false);
    expect(params?.get("sort_by")).toBe("group_name");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("drops expanded names that no longer match a routing group", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderRetainingMountWrites("?expanded=ghost-group,prod-group", onUrlUpdate);

    expect(await screen.findAllByText(USAGE_PANEL_TITLE)).toHaveLength(1);
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("expanded")).toBe("prod-group");
  });

  it("keeps an expanded group that is only hidden by the search", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderRetainingMountWrites("?expanded=prod-group&group_search=dev", onUrlUpdate);

    expect(visibleGroupNames()).toEqual(["dev-group"]);
    expect(screen.queryByText(USAGE_PANEL_TITLE)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear search" }));

    expect(await screen.findAllByText(USAGE_PANEL_TITLE)).toHaveLength(1);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("expanded")).toBe("prod-group");
  });

  it("leaves expanded names alone until the routing groups have loaded", async () => {
    vi.mocked(useRoutingGroups).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: vi.fn(),
      isFetching: true,
    } as unknown as ReturnType<typeof useRoutingGroups>);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderRetainingMountWrites("?expanded=prod-group", onUrlUpdate);

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
