import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  usePolicyVersions,
  useCreatePolicyVersion,
  useUpdatePolicyVersionStatus,
} from "./usePolicyVersions";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockListPolicyVersions = vi.fn();
const mockCreatePolicyVersion = vi.fn();
const mockUpdatePolicyVersionStatus = vi.fn();

vi.mock("@/components/networking", () => ({
  listPolicyVersions: (...args: unknown[]) => mockListPolicyVersions(...args),
  createPolicyVersion: (...args: unknown[]) => mockCreatePolicyVersion(...args),
  updatePolicyVersionStatus: (...args: unknown[]) =>
    mockUpdatePolicyVersionStatus(...args),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

// Import the mocked module to assert on it
import NotificationsManager from "@/components/molecules/notifications_manager";

const mockUseAuthorized = vi.fn();
vi.mock("../useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// ── Shared test helpers ──────────────────────────────────────────────────────

function createTestEnv() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  vi.clearAllMocks();

  mockUseAuthorized.mockReturnValue({
    accessToken: "test-access-token",
    userRole: "Admin",
    userId: "test-user-id",
    token: "test-token",
    userEmail: "test@example.com",
    premiumUser: false,
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  return { queryClient, wrapper };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("usePolicyVersions", () => {
  let env: ReturnType<typeof createTestEnv>;

  beforeEach(() => {
    env = createTestEnv();
  });

  it("fetches versions when policyName is provided", async () => {
    const mockResponse = {
      policy_name: "my-policy",
      versions: [
        { policy_id: "v1", policy_name: "my-policy", version_number: 1, version_status: "production" },
        { policy_id: "v2", policy_name: "my-policy", version_number: 2, version_status: "draft" },
      ],
      total_count: 2,
    };
    mockListPolicyVersions.mockResolvedValue(mockResponse);

    const { result } = renderHook(
      () => usePolicyVersions({ policyName: "my-policy" }),
      { wrapper: env.wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(mockListPolicyVersions).toHaveBeenCalledWith("test-access-token", "my-policy");
    expect(result.current.data?.versions).toHaveLength(2);
    expect(result.current.data?.versions[0].policy_id).toBe("v1");
  });

  it("does not fetch when policyName is null", () => {
    const { result } = renderHook(
      () => usePolicyVersions({ policyName: null }),
      { wrapper: env.wrapper }
    );

    expect(result.current.fetchStatus).toBe("idle");
    // isLoading (not isPending) must be false when query is disabled
    expect(result.current.isLoading).toBe(false);
    expect(mockListPolicyVersions).not.toHaveBeenCalled();
  });

  it("does not fetch when enabled is false", () => {
    const { result } = renderHook(
      () => usePolicyVersions({ policyName: "my-policy", enabled: false }),
      { wrapper: env.wrapper }
    );

    expect(result.current.fetchStatus).toBe("idle");
    // isLoading (not isPending) must be false when query is disabled
    expect(result.current.isLoading).toBe(false);
    expect(mockListPolicyVersions).not.toHaveBeenCalled();
  });

  it("defaults versions to empty array when response has undefined versions", async () => {
    mockListPolicyVersions.mockResolvedValue({
      policy_name: "my-policy",
      versions: undefined,
      total_count: 0,
    });

    const { result } = renderHook(
      () => usePolicyVersions({ policyName: "my-policy" }),
      { wrapper: env.wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.versions).toEqual([]);
  });
});

describe("useCreatePolicyVersion", () => {
  let env: ReturnType<typeof createTestEnv>;

  beforeEach(() => {
    env = createTestEnv();
  });

  it("calls createPolicyVersion and shows success notification", async () => {
    const newPolicy = { policy_id: "v3", policy_name: "my-policy", version_number: 3 };
    mockCreatePolicyVersion.mockResolvedValue(newPolicy);

    const { result } = renderHook(
      () => useCreatePolicyVersion("my-policy"),
      { wrapper: env.wrapper }
    );

    const returned = await result.current.mutateAsync();

    expect(mockCreatePolicyVersion).toHaveBeenCalledWith("test-access-token", "my-policy");
    expect(returned).toEqual(newPolicy);
    expect(NotificationsManager.success).toHaveBeenCalledWith("New draft version created");
  });

  it("invalidates the versions cache on success", async () => {
    mockCreatePolicyVersion.mockResolvedValue({ policy_id: "v3" });
    const invalidateSpy = vi.spyOn(env.queryClient, "invalidateQueries");

    const { result } = renderHook(
      () => useCreatePolicyVersion("my-policy"),
      { wrapper: env.wrapper }
    );

    await result.current.mutateAsync();

    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ["policyVersions", "detail", "my-policy"],
    });
  });

  it("shows error notification on failure", async () => {
    mockCreatePolicyVersion.mockRejectedValue(new Error("Server error"));

    const { result } = renderHook(
      () => useCreatePolicyVersion("my-policy"),
      { wrapper: env.wrapper }
    );

    await expect(result.current.mutateAsync()).rejects.toThrow("Server error");
    expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
      "Failed to create version: Server error"
    );
  });

  it("does not show user-facing notification when policyName is null", async () => {
    const { result } = renderHook(
      () => useCreatePolicyVersion(null),
      { wrapper: env.wrapper }
    );

    await expect(result.current.mutateAsync()).rejects.toThrow(
      "Missing access token or policy name"
    );
    expect(NotificationsManager.fromBackend).not.toHaveBeenCalled();
  });
});

describe("useUpdatePolicyVersionStatus", () => {
  let env: ReturnType<typeof createTestEnv>;

  beforeEach(() => {
    env = createTestEnv();
  });

  it("publishes a version and shows success notification", async () => {
    const updatedPolicy = { policy_id: "v2", version_status: "published" };
    mockUpdatePolicyVersionStatus.mockResolvedValue(updatedPolicy);

    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus("my-policy"),
      { wrapper: env.wrapper }
    );

    const returned = await result.current.mutateAsync({
      policyId: "v2",
      status: "published",
    });

    expect(mockUpdatePolicyVersionStatus).toHaveBeenCalledWith(
      "test-access-token",
      "v2",
      "published"
    );
    expect(returned).toEqual(updatedPolicy);
    expect(NotificationsManager.success).toHaveBeenCalledWith(
      expect.stringContaining("Version published")
    );
  });

  it("invalidates the versions cache on success", async () => {
    mockUpdatePolicyVersionStatus.mockResolvedValue({ policy_id: "v2" });
    const invalidateSpy = vi.spyOn(env.queryClient, "invalidateQueries");

    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus("my-policy"),
      { wrapper: env.wrapper }
    );

    await result.current.mutateAsync({ policyId: "v2", status: "published" });

    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ["policyVersions", "detail", "my-policy"],
    });
  });

  it("promotes to production and shows success notification", async () => {
    const updatedPolicy = { policy_id: "v2", version_status: "production" };
    mockUpdatePolicyVersionStatus.mockResolvedValue(updatedPolicy);

    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus("my-policy"),
      { wrapper: env.wrapper }
    );

    await result.current.mutateAsync({
      policyId: "v2",
      status: "production",
    });

    expect(mockUpdatePolicyVersionStatus).toHaveBeenCalledWith(
      "test-access-token",
      "v2",
      "production"
    );
    expect(NotificationsManager.success).toHaveBeenCalledWith(
      "Version promoted to production"
    );
  });

  it("shows error notification on publish failure", async () => {
    mockUpdatePolicyVersionStatus.mockRejectedValue(new Error("Forbidden"));

    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus("my-policy"),
      { wrapper: env.wrapper }
    );

    await expect(
      result.current.mutateAsync({ policyId: "v2", status: "published" })
    ).rejects.toThrow("Forbidden");
    expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
      "Failed to publish: Forbidden"
    );
  });

  it("shows error notification on promote failure", async () => {
    mockUpdatePolicyVersionStatus.mockRejectedValue(new Error("Not found"));

    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus("my-policy"),
      { wrapper: env.wrapper }
    );

    await expect(
      result.current.mutateAsync({ policyId: "v2", status: "production" })
    ).rejects.toThrow("Not found");
    expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
      "Failed to promote to production: Not found"
    );
  });

  it("does not show user-facing notification when policyName is null", async () => {
    const { result } = renderHook(
      () => useUpdatePolicyVersionStatus(null),
      { wrapper: env.wrapper }
    );

    await expect(
      result.current.mutateAsync({ policyId: "v2", status: "published" })
    ).rejects.toThrow("Missing access token or policy name");
    expect(NotificationsManager.fromBackend).not.toHaveBeenCalled();
  });
});
