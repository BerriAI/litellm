import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, testQueryClient } from "../../../tests/test-utils";
import { backgroundFetchClient } from "@/lib/http/api";
import { useDeprecationAlerts } from "./useDeprecationAlerts";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";

vi.mock("@/lib/http/api", () => ({ backgroundFetchClient: { GET: vi.fn() } }));
vi.mock("@/app/(dashboard)/hooks/currentUser/useCurrentUser", () => ({ useCurrentUser: vi.fn() }));

function AlertsHost() {
  const { data, isError, fetchStatus } = useDeprecationAlerts("test-session");
  if (isError) return <div>Unavailable</div>;
  if (fetchStatus !== "idle") return <div>Loading</div>;
  return <div>{data?.length ?? 0} alerts</div>;
}

describe("deprecation alert API wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
  });
  it("stops loading usage and acknowledgements when model deprecations are disabled", async () => {
    vi.mocked(useCurrentUser).mockReturnValue({ data: { user_id: "admin", user_role: "proxy_admin" } } as ReturnType<
      typeof useCurrentUser
    >);
    vi.mocked(backgroundFetchClient.GET).mockResolvedValue({ data: { model_deprecations_enabled: false } } as never);
    renderWithProviders(<AlertsHost />);
    expect(await screen.findByText("0 alerts")).toBeInTheDocument();
    expect(backgroundFetchClient.GET).toHaveBeenCalledExactlyOnceWith("/alerts/monitoring", expect.anything());
  });
  it("should avoid querying gateway-wide usage for non-admins", () => {
    vi.mocked(useCurrentUser).mockReturnValue({
      data: { user_id: "internal", user_role: "internal_user" },
    } as ReturnType<typeof useCurrentUser>);
    renderWithProviders(<AlertsHost />);
    expect(screen.getByText("0 alerts")).toBeInTheDocument();
    expect(backgroundFetchClient.GET).not.toHaveBeenCalled();
  });
  it("should skip the usage query when no configured models need attention", async () => {
    vi.mocked(useCurrentUser).mockReturnValue({ data: { user_id: "admin", user_role: "proxy_admin" } } as ReturnType<
      typeof useCurrentUser
    >);
    vi.mocked(backgroundFetchClient.GET).mockResolvedValue({
      data: { deprecated: [], imminent: [], upcoming: [] },
    } as never);
    renderWithProviders(<AlertsHost />);
    expect(await screen.findByText("0 alerts")).toBeInTheDocument();
    expect(backgroundFetchClient.GET).toHaveBeenCalledTimes(2);
    expect(backgroundFetchClient.GET).toHaveBeenCalledWith(
      "/model/deprecations",
      expect.objectContaining({ params: { query: { warn_within_days: 30 } } }),
    );
  });
  it("should load acknowledgement state from the alerting API for models with observed usage", async () => {
    vi.mocked(useCurrentUser).mockReturnValue({ data: { user_id: "admin", user_role: "proxy_admin" } } as ReturnType<
      typeof useCurrentUser
    >);
    const model = {
      model_name: "support",
      litellm_model: "anthropic/claude-sonnet-4-5",
      deprecation_date: "2026-11-30",
      status: "imminent",
    };
    const schedule = { deprecated: [], imminent: [model], upcoming: [] };
    const activity = {
      results: [
        { date: "2026-11-04", breakdown: { models: { [model.litellm_model]: { metrics: { api_requests: 10 } } } } },
      ],
    };
    vi.mocked(backgroundFetchClient.GET)
      .mockResolvedValueOnce({ data: { model_deprecations_enabled: true } } as never)
      .mockResolvedValueOnce({ data: schedule } as never)
      .mockResolvedValueOnce({ data: activity } as never)
      .mockResolvedValueOnce({ data: { status: "open" } } as never);
    renderWithProviders(<AlertsHost />);
    expect(await screen.findByText("1 alerts")).toBeInTheDocument();
    expect(backgroundFetchClient.GET).toHaveBeenLastCalledWith(
      "/alerting/model-deprecations/state",
      expect.objectContaining({
        params: { query: { model: model.litellm_model, deprecation_date: model.deprecation_date } },
      }),
    );
  });
});
