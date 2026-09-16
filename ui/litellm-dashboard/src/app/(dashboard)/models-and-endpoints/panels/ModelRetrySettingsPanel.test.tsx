import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { chooseSelectOption, render, screen, waitFor } from "@/../tests/test-utils";

import ModelRetrySettingsPanel from "./ModelRetrySettingsPanel";

const { getCallbacksCall, dashboardData } = vi.hoisted(() => ({
  getCallbacksCall: vi.fn(),
  dashboardData: vi.fn(),
}));

vi.mock("@/components/networking", () => ({ getCallbacksCall }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userId: "u-admin", userRole: "Admin" }),
}));

vi.mock("@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy", () => ({
  useUpdateRetryPolicy: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/useModelDashboardData", () => ({
  useModelDashboardData: () => dashboardData(),
}));

const MODEL_GROUPS = ["claude-opus", "gpt-5"];

const loadedGroups = () => ({ availableModelGroups: MODEL_GROUPS, isLoading: false });
const loadingGroups = () => ({ availableModelGroups: [], isLoading: true });

const renderPanel = (searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  render(<ModelRetrySettingsPanel />, {
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

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => {
  const lastCall = onUrlUpdate.mock.calls.at(-1);
  if (!lastCall) throw new Error("expected a URL update");
  return lastCall[0].searchParams;
};

describe("ModelRetrySettingsPanel retry_scope", () => {
  beforeEach(() => {
    getCallbacksCall.mockReset();
    getCallbacksCall.mockResolvedValue({ router_settings: { num_retries: 2 } });
    dashboardData.mockReset();
    dashboardData.mockImplementation(loadedGroups);
  });

  it("shows the global policy when the URL names no scope", () => {
    renderPanel();

    expect(screen.getByRole("heading", { name: "Global Retry Policy" })).toBeInTheDocument();
  });

  it("shows the model group named in the URL", () => {
    renderPanel("?retry_scope=gpt-5");

    expect(screen.getByRole("heading", { name: "Retry Policy for gpt-5" })).toBeInTheDocument();
  });

  it("keeps a model group scope in the URL while the model groups are still loading", async () => {
    dashboardData.mockImplementation(loadingGroups);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { rerender } = renderPanel("?retry_scope=gpt-5", onUrlUpdate);

    expect(screen.getByRole("heading", { name: "Retry Policy for gpt-5" })).toBeInTheDocument();

    dashboardData.mockImplementation(loadedGroups);
    rerender(<ModelRetrySettingsPanel />);

    expect(screen.getByRole("heading", { name: "Retry Policy for gpt-5" })).toBeInTheDocument();
    await waitFor(() => expect(getCallbacksCall).toHaveBeenCalled());
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("falls back to the global policy and drops a model group that no longer exists", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("?retry_scope=retired-model&tab=retry-settings", onUrlUpdate);

    expect(screen.getByRole("heading", { name: "Global Retry Policy" })).toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).has("retry_scope")).toBe(false);
    expect(lastUrl(onUrlUpdate).get("tab")).toBe("retry-settings");
  });

  it("writes the selected scope to the URL and clears it for the global default", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);

    await chooseSelectOption(user, screen.getByRole("combobox", { name: /retry policy scope/i }), "gpt-5");

    await waitFor(() => expect(lastUrl(onUrlUpdate).get("retry_scope")).toBe("gpt-5"));
    expect(screen.getByRole("heading", { name: "Retry Policy for gpt-5" })).toBeInTheDocument();

    await chooseSelectOption(user, screen.getByRole("combobox", { name: /retry policy scope/i }), "Global Default");

    await waitFor(() => expect(lastUrl(onUrlUpdate).has("retry_scope")).toBe(false));
    expect(screen.getByRole("heading", { name: "Global Retry Policy" })).toBeInTheDocument();
  });
});
