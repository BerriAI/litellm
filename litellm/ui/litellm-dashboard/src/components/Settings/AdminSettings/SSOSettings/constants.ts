import type { ParseKeys, TFunction } from "i18next";
import googleLogo from "../../../../../public/assets/logos/google.svg";
import microsoftAzureLogo from "../../../../../public/assets/logos/microsoft_azure.svg";

// SSO Provider logos
export const ssoProviderLogoMap: Record<string, string> = {
  google: googleLogo.src,
  microsoft: microsoftAzureLogo.src,
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
  saml: "",
};

// SSO Provider display names (consistent between select dropdown and table)
export const ssoProviderDisplayNames: Record<string, (t: TFunction) => string> = {
  google: () => "Google SSO",
  microsoft: () => "Microsoft SSO",
  okta: () => "Okta / Auth0 SSO",
  generic: (t) => t("adminPanel.ssoSettings.providers.generic"),
  saml: () => "SAML SSO",
};

export const defaultRoleDisplayNames: Record<string, ParseKeys> = {
  internal_user_viewer: "adminPanel.ssoSettings.roles.internalUserViewer",
  internal_user: "adminPanel.ssoSettings.roles.internalUser",
  proxy_admin_viewer: "adminPanel.ssoSettings.roles.proxyAdminViewer",
  proxy_admin: "adminPanel.ssoSettings.roles.proxyAdmin",
};
