"use client";

import { useTranslation } from "react-i18next";

import { useUpstreamProxies } from "@/app/(dashboard)/hooks/upstreamProxies/useUpstreamProxies";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { all_admin_roles } from "@/utils/roles";

import { upstreamProxyAddress } from "./api";

export const UPSTREAM_PROXY_FIELD = "upstream_proxy";

interface UpstreamProxySelectProps {
  id?: string;
  value: string | null | undefined;
  onChange: (value: string | null) => void;
}

export function UpstreamProxySelect({ id, value, onChange }: UpstreamProxySelectProps) {
  const { t } = useTranslation();
  const { data } = useUpstreamProxies();
  const options = (data?.proxies ?? []).map((proxy) => ({
    label: proxy.name,
    value: proxy.name,
    sublabel: upstreamProxyAddress(proxy),
  }));
  return (
    <SearchSelect
      inputId={id}
      aria-label={t("upstreamProxies.fieldLabel")}
      options={options}
      value={value ?? ""}
      placeholder={t("upstreamProxies.environmentDefault")}
      emptyText={t("upstreamProxies.noProxies")}
      allowClear
      onValueChange={(selected) => onChange(selected ? selected : null)}
    />
  );
}

interface MountedUpstreamProxyFieldProps {
  className?: string;
}

export function MountedUpstreamProxyField({ className = "mb-4" }: MountedUpstreamProxyFieldProps) {
  const { t } = useTranslation();
  const { userRole } = useAuthorized();
  if (!all_admin_roles.includes(userRole ?? "")) {
    return null;
  }
  return (
    <MountedFormField
      name={UPSTREAM_PROXY_FIELD}
      label={labelWithHint(t("upstreamProxies.fieldLabel"), t("upstreamProxies.fieldHint"))}
      className={className}
    >
      {(control) => (
        <UpstreamProxySelect
          id={control.id}
          value={control.value as string | null | undefined}
          onChange={control.onChange}
        />
      )}
    </MountedFormField>
  );
}
