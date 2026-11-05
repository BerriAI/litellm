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
