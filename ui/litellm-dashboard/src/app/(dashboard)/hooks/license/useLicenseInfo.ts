import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { getLicenseInfo, LicenseInfo } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { createQueryKeys } from "../common/queryKeysFactory";

const licenseInfoKeys = createQueryKeys("licenseInfo");

export const useLicenseInfo = (
  accessToken: string | null | undefined,
  userRole: string | null | undefined,
): UseQueryResult<LicenseInfo | null> => {
  const options = {
    queryKey: licenseInfoKeys.detail("license"),
    queryFn: () => getLicenseInfo(accessToken!),
    enabled: Boolean(accessToken) && hasCapability(userRole, "viewLicenseInfo"),
    staleTime: 5 * 60 * 1000,
    retry: false,
  };
  return useQuery<LicenseInfo | null>(options);
};
