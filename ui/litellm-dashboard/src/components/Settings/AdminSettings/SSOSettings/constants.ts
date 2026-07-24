import googleLogo from "../../../../../public/assets/logos/google.svg";
import microsoftAzureLogo from "../../../../../public/assets/logos/microsoft_azure.svg";

// SSO Provider logos
export const ssoProviderLogoMap: Record<string, string> = {
  google: googleLogo.src,
  microsoft: microsoftAzureLogo.src,
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
};

// SSO Provider display names (consistent between select dropdown and table)
export const ssoProviderDisplayNames: Record<string, string> = {
  google: "Google SSO",
  microsoft: "Microsoft SSO",
  okta: "Okta / Auth0 SSO",
  generic: "Generic SSO",
};

export const defaultRoleDisplayNames: Record<string, string> = {
  internal_user_viewer: "Internal Viewer",
  internal_user: "Internal User",
  proxy_admin_viewer: "Proxy Admin Viewer",
  proxy_admin: "Proxy Admin",
};
