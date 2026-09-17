export const ORGANIZATION_TABS = ["overview", "members", "settings"] as const;
export type OrganizationTab = (typeof ORGANIZATION_TABS)[number];
export const ORGANIZATION_TAB_URL_KEY = "org_tab";
