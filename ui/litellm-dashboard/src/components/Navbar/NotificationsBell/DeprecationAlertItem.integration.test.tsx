import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient } from "../../../../tests/test-utils";
import { backgroundFetchClient } from "@/lib/http/api";
import DeprecationAlertItem from "./DeprecationAlertItem";
import { useDeprecationAlerts } from "./useDeprecationAlerts";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";

vi.mock("@/lib/http/api", () => ({ backgroundFetchClient: { GET: vi.fn() } }));
vi.mock("@/app/(dashboard)/hooks/currentUser/useCurrentUser", () => ({ useCurrentUser: vi.fn() }));

const alert = {
  id: "sonnet",
  usageModel: "anthropic/claude-sonnet-4-5",
  requests: 123,
  lastUsedDate: "2026-11-04",
  model: {
    model_name: "support",
    deprecation_date: "2026-11-30",
    days_until_deprecation: 26,
    status: "imminent" as const,
  },
};
const range = { start_date: "2026-10-06", end_date: "2026-11-04" };
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
  it("should render the alert before fetching key details", () => {
    renderWithProviders(
      <DeprecationAlertItem alert={alert} range={range} scope="demo" dismissed={false} onDismiss={vi.fn()} />,
    );
    expect(screen.getByRole("article", { name: /retires in 26 days/ })).toHaveTextContent("123 requests");
    expect(backgroundFetchClient.GET).not.toHaveBeenCalled();
  });
  it("should fetch affected keys for the underlying model and link to the matching key and team", async () => {
    vi.mocked(backgroundFetchClient.GET).mockResolvedValue({
      data: {
        api_keys: [
          {
            api_key: "reporting-key-1",
            metadata: { key_alias: "support-production", team_id: "support-team" },
            metrics: { api_requests: 123 },
          },
        ],
      },
    } as never);
    const user = userEvent.setup();
    renderWithProviders(
      <DeprecationAlertItem alert={alert} range={range} scope="demo" dismissed={false} onDismiss={vi.fn()} />,
    );
    await user.click(screen.getByRole("button", { name: "Review affected keys" }));
    expect(await screen.findByRole("link", { name: "support-production" })).toHaveAttribute(
      "href",
      "/ui/api-keys?key=reporting-key-1",
    );
    expect(screen.getByRole("link", { name: "Team: support-team" })).toHaveAttribute(
      "href",
      "/ui/teams?team=support-team",
    );
    expect(backgroundFetchClient.GET).toHaveBeenCalledWith(
      "/user/daily/activity/aggregated/model_top_keys",
      expect.objectContaining({
        params: { query: { ...range, model_group: alert.usageModel, by_model_group: false, limit: 10, timezone: 0 } },
      }),
    );
  });
  it("should offer retry when affected keys cannot load", async () => {
    vi.mocked(backgroundFetchClient.GET).mockRejectedValue(new Error("Unavailable"));
    const user = userEvent.setup();
    renderWithProviders(
      <DeprecationAlertItem alert={alert} range={range} scope="demo" dismissed={false} onDismiss={vi.fn()} />,
    );
    await user.click(screen.getByRole("button", { name: "Review affected keys" }));
    expect(await screen.findByText("Could not load affected keys.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
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
    expect(backgroundFetchClient.GET).toHaveBeenCalledTimes(1);
    expect(backgroundFetchClient.GET).toHaveBeenCalledWith(
      "/model/deprecations",
      expect.objectContaining({ params: { query: { warn_within_days: 30 } } }),
    );
  });
});
