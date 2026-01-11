// SSO Provider logos
export const ssoProviderLogoMap: Record<string, string> = {
  google: "https://artificialanalysis.ai/img/logos/google_small.svg",
  microsoft: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
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
