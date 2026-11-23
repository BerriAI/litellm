import { getProxyBaseUrl, getUiPageUrl } from "@/components/networking";
import { buildLoginUrlWithReturn, getLoginUrl, storeReturnUrl } from "@/utils/returnUrlUtils";

export const redirectToLogin = () => {
  storeReturnUrl();
  const baseLoginUrl = getUiPageUrl(getLoginUrl(getProxyBaseUrl()));
  window.location.replace(buildLoginUrlWithReturn(baseLoginUrl));
};
