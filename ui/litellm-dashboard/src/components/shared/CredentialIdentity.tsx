import type { ReactNode } from "react";

import type { CredentialItem } from "@/components/networking";
import { credentialLabelsByName } from "@/components/shared/credentialOptions";
import { cn } from "@/lib/cva.config";

interface CredentialIdentityProps {
  credentialName: string;
  label: string | undefined;
  icon?: ReactNode;
  className?: string;
}

export function CredentialIdentity({ credentialName, label, icon, className }: CredentialIdentityProps) {
  const shown = label ?? credentialName;
  return (
    <span className={cn("flex min-w-0 flex-col gap-0.5", className)} title={credentialName}>
      <span className="flex min-w-0 items-center gap-1.5">
        {icon}
        <span className="truncate">{shown}</span>
      </span>
      {shown !== credentialName && (
        <span className="truncate font-mono text-xs font-normal text-muted-foreground">{credentialName}</span>
      )}
    </span>
  );
}

interface AttachedCredentialProps {
  credentialName: string | null | undefined;
  credentials: CredentialItem[];
}

export function AttachedCredential({ credentialName, credentials }: AttachedCredentialProps) {
  if (!credentialName) {
    return <>Manual</>;
  }
  return (
    <CredentialIdentity
      credentialName={credentialName}
      label={credentialLabelsByName(credentials).get(credentialName)}
    />
  );
}

export function CredentialNameHint({ credentialName, credentials }: AttachedCredentialProps) {
  const label = credentialName ? credentialLabelsByName(credentials).get(credentialName) : undefined;
  if (!credentialName || label === undefined || label === credentialName) {
    return null;
  }
  return <p className="mt-1 truncate font-mono text-xs text-muted-foreground">{credentialName}</p>;
}
