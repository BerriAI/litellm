export interface MenuItem {
  key: string;
  page: string;
  route?: string;
  label: React.ReactNode;
  untranslatedLabel?: string;
  roles?: string[];
  children?: MenuItem[];
  icon?: React.ReactNode;
  external_url?: string;
}

export interface MenuGroup {
  groupLabel: string;
  items: MenuItem[];
  roles?: string[];
}
