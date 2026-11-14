import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

const setup = (overrides: { mutateAsync?: ReturnType<typeof vi.fn>; isPending?: boolean } = {}) => {
  const mutateAsync = overrides.mutateAsync ?? vi.fn().mockResolvedValue(undefined);
  vi.mocked(useRoutingGroups).mockReturnValue({
    data: { routingGroups: [prodGroup, devGroup], availableStrategies: [] },
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
    render(<RoutingGroups />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Delete routing group?")).not.toBeInTheDocument();
  });

  it("should name the group being deleted in the confirmation", async () => {
    const user = userEvent.setup();
    setup();
    render(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);

    expect(within(dialog).getByText("Delete routing group?")).toBeInTheDocument();
    expect(within(dialog).getByText("prod-group")).toBeInTheDocument();
    expect(within(dialog).getByText(/This cannot be undone/)).toBeInTheDocument();
  });

  it("should save the remaining groups and report success when confirmed", async () => {
    const user = userEvent.setup();
    const { mutateAsync } = setup();
    render(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    expect(mutateAsync).toHaveBeenCalledWith([devGroup]);
    expect(toast.success).toHaveBeenCalledWith('Deleted routing group "prod-group"');
  });

  it("should report the failure and keep the confirmation open when the save rejects", async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockRejectedValue(new Error("boom"));
    setup({ mutateAsync });
    render(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    expect(toast.error).toHaveBeenCalledWith("boom");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("should dismiss without saving when cancelled", async () => {
    const user = userEvent.setup();
    const { mutateAsync } = setup();
    render(<RoutingGroups />);

    const dialog = await openDeleteConfirm(user);
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(mutateAsync).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
