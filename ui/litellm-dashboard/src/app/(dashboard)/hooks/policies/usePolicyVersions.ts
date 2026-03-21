import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import useAuthorized from "../useAuthorized";
import { createQueryKeys } from "../common/queryKeysFactory";
import {
  listPolicyVersions,
  createPolicyVersion,
  updatePolicyVersionStatus,
} from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Policy } from "@/components/policies/types";

// ── Query keys ──────────────────────────────────────────────────────────────

export const policyVersionKeys = createQueryKeys("policyVersions");

// ── Types ───────────────────────────────────────────────────────────────────

export interface PolicyVersionsResponse {
  policy_name: string;
  versions: Policy[] | undefined;
  total_count: number;
}

/** Output type after `select` normalizes the response — versions is always defined. */
export interface PolicyVersionsData {
  policy_name: string;
  versions: Policy[];
  total_count: number;
}

// ── Hook ────────────────────────────────────────────────────────────────────

export interface UsePolicyVersionsOptions {
  policyName: string | null | undefined;
  enabled?: boolean;
}

/** Stable key used when the query is disabled to avoid undefined in cache keys. */
const DISABLED_POLICY_KEY = "__disabled__";

export const usePolicyVersions = ({
  policyName,
  enabled = true,
}: UsePolicyVersionsOptions) => {
  const { accessToken } = useAuthorized();
  const isEnabled = Boolean(accessToken && policyName && enabled);

  return useQuery<PolicyVersionsResponse, Error, PolicyVersionsData>({
    queryKey: policyVersionKeys.detail(policyName ?? DISABLED_POLICY_KEY),
    queryFn: async () =>
      await listPolicyVersions(accessToken!, policyName!) as PolicyVersionsResponse,
    enabled: isEnabled,
    select: (data) => ({
      ...data,
      versions: data.versions ?? [],
    }),
  });
};

// ── Mutations ───────────────────────────────────────────────────────────────

export const useCreatePolicyVersion = (policyName: string | null | undefined) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<Policy, Error>({
    mutationFn: async () => {
      if (!accessToken || !policyName) {
        throw new Error("Missing access token or policy name");
      }
      return await createPolicyVersion(accessToken, policyName);
    },
    onSuccess: () => {
      NotificationsManager.success("New draft version created");
      if (policyName) {
        queryClient.invalidateQueries({
          queryKey: policyVersionKeys.detail(policyName),
        });
      }
    },
    onError: (error) => {
      if (error.message === "Missing access token or policy name") return;
      NotificationsManager.fromBackend(
        "Failed to create version: " + error.message
      );
    },
  });
};

export const useUpdatePolicyVersionStatus = (
  policyName: string | null | undefined
) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<
    Policy,
    Error,
    { policyId: string; status: "published" | "production" }
  >({
    mutationFn: async ({ policyId, status }) => {
      if (!accessToken || !policyName) {
        throw new Error("Missing access token or policy name");
      }
      return await updatePolicyVersionStatus(accessToken, policyId, status);
    },
    onSuccess: (_data, variables) => {
      const label =
        variables.status === "published"
          ? "Version published. You can test it in the Playground by selecting this version in the Policies dropdown."
          : "Version promoted to production";
      NotificationsManager.success(label);
      if (policyName) {
        queryClient.invalidateQueries({
          queryKey: policyVersionKeys.detail(policyName),
        });
      }
    },
    onError: (error, variables) => {
      if (error.message === "Missing access token or policy name") return;
      const action =
        variables.status === "published" ? "publish" : "promote to production";
      NotificationsManager.fromBackend(
        `Failed to ${action}: ${error.message}`
      );
    },
  });
};
