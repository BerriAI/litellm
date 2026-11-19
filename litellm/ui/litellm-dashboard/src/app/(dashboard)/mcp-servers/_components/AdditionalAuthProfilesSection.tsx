import type { ParseKeys } from "i18next";
import { Plus, Trash2 } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PasswordInput } from "@/components/shared/PasswordInput";
import {
  PROFILE_KIND,
  PROFILE_KIND_LABEL_KEYS,
  PROFILE_KINDS,
  TOKEN_ENDPOINT_AUTH_METHODS,
  newProfileDraft,
  profileProblem,
  type AdditionalAuthDraft,
  type ProfileKind,
  type ProfileProblem,
} from "./additionalAuthProfiles";

const UNSET_METHOD = "unset";

const PROBLEM_KEYS: Readonly<Record<ProfileProblem, ParseKeys>> = {
  id_format: "mcpServers.additionalAuth.problems.idFormat",
  id_reserved: "mcpServers.additionalAuth.problems.idReserved",
  id_duplicate: "mcpServers.additionalAuth.problems.idDuplicate",
  missing_field: "mcpServers.additionalAuth.problems.missingField",
};

interface AdditionalAuthProfilesSectionProps {
  readonly value: readonly AdditionalAuthDraft[];
  readonly onChange: (next: readonly AdditionalAuthDraft[]) => void;
}

interface ProfileRowProps {
  readonly draft: AdditionalAuthDraft;
  readonly drafts: readonly AdditionalAuthDraft[];
  readonly onChange: (next: AdditionalAuthDraft) => void;
  readonly onRemove: () => void;
}

const ProfileField: React.FC<{ htmlFor: string; label: string; children: React.ReactNode }> = ({
  htmlFor,
  label,
  children,
}) => (
  <div className="space-y-1">
    <Label htmlFor={htmlFor}>{label}</Label>
    {children}
  </div>
);

const ProfileRow: React.FC<ProfileRowProps> = ({ draft, drafts, onChange, onRemove }) => {
  const { t } = useTranslation();
  const rowId = React.useId();
  const problem = profileProblem(draft, drafts);
  const set = (field: keyof AdditionalAuthDraft) => (event: React.ChangeEvent<HTMLInputElement>) =>
    onChange({ ...draft, [field]: event.target.value });
  const secretPlaceholder = draft.isNew
    ? t("mcpServers.additionalAuth.secretPlaceholder")
    : t("mcpServers.additionalAuth.secretPlaceholderEditing");
  const kindItems = PROFILE_KINDS.map((kind) => ({ value: kind, label: t(PROFILE_KIND_LABEL_KEYS[kind]) }));
  const methodItems = [
    { value: UNSET_METHOD, label: t("mcpServers.additionalAuth.tokenEndpointAuthMethodUnset") },
    ...TOKEN_ENDPOINT_AUTH_METHODS.map((method) => ({ value: method, label: method })),
  ];

  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-4" data-slot="additional-auth-profile">
      <div className="flex items-end gap-3">
        <div className="flex-1">
          <ProfileField htmlFor={`${rowId}-id`} label={t("mcpServers.additionalAuth.id")}>
            <Input
              id={`${rowId}-id`}
              value={draft.id}
              onChange={set("id")}
              disabled={!draft.isNew}
              placeholder="tickets-bots"
              aria-invalid={problem !== null && problem !== "missing_field"}
              className="font-mono"
            />
          </ProfileField>
        </div>
        <div className="w-56">
          <ProfileField htmlFor={`${rowId}-kind`} label={t("mcpServers.additionalAuth.kind")}>
            <Select
              value={draft.kind}
              onValueChange={(kind) => onChange({ ...draft, kind: kind as ProfileKind })}
              items={kindItems}
              disabled={!draft.isNew}
            >
              <SelectTrigger id={`${rowId}-kind`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {kindItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ProfileField>
        </div>
        <Button variant="ghost" size="icon" aria-label={t("mcpServers.additionalAuth.remove")} onClick={onRemove}>
          <Trash2 />
        </Button>
      </div>
      {draft.kind === PROFILE_KIND.CLIENT_CREDENTIALS ? (
        <>
          <ProfileField htmlFor={`${rowId}-client_id`} label={t("mcpServers.additionalAuth.clientId")}>
            <Input
              id={`${rowId}-client_id`}
              value={draft.client_id}
              onChange={set("client_id")}
              placeholder={draft.isNew ? undefined : t("mcpServers.additionalAuth.secretPlaceholderEditing")}
            />
          </ProfileField>
          <ProfileField htmlFor={`${rowId}-client_secret`} label={t("mcpServers.additionalAuth.clientSecret")}>
            <PasswordInput
              id={`${rowId}-client_secret`}
              value={draft.client_secret}
              onChange={set("client_secret")}
              placeholder={secretPlaceholder}
            />
          </ProfileField>
          <ProfileField htmlFor={`${rowId}-token_url`} label={t("mcpServers.additionalAuth.tokenUrl")}>
            <Input
              id={`${rowId}-token_url`}
              value={draft.token_url}
              onChange={set("token_url")}
              placeholder="https://auth.example.com/oauth/token"
            />
          </ProfileField>
          <ProfileField htmlFor={`${rowId}-scopes`} label={t("mcpServers.additionalAuth.scopes")}>
            <Input
              id={`${rowId}-scopes`}
              value={draft.scopes}
              onChange={set("scopes")}
              placeholder={t("mcpServers.additionalAuth.scopesPlaceholder")}
            />
          </ProfileField>
          <ProfileField
            htmlFor={`${rowId}-token_endpoint_auth_method`}
            label={t("mcpServers.additionalAuth.tokenEndpointAuthMethod")}
          >
            <Select
              value={draft.token_endpoint_auth_method || UNSET_METHOD}
              onValueChange={(method) =>
                onChange({
                  ...draft,
                  token_endpoint_auth_method: method === null || method === UNSET_METHOD ? "" : method,
                })
              }
              items={methodItems}
            >
              <SelectTrigger id={`${rowId}-token_endpoint_auth_method`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {methodItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ProfileField>
        </>
      ) : (
        <ProfileField htmlFor={`${rowId}-authentication_token`} label={t("mcpServers.additionalAuth.token")}>
          <PasswordInput
            id={`${rowId}-authentication_token`}
            value={draft.authentication_token}
            onChange={set("authentication_token")}
            placeholder={secretPlaceholder}
          />
        </ProfileField>
      )}
      {problem !== null && <p className="text-sm text-destructive">{t(PROBLEM_KEYS[problem])}</p>}
    </div>
  );
};

const AdditionalAuthProfilesSection: React.FC<AdditionalAuthProfilesSectionProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <div className="space-y-3 rounded-lg border border-border bg-muted p-4">
      <div>
        <strong className="text-sm font-semibold">{t("mcpServers.additionalAuth.title")}</strong>
        <p className="text-xs text-muted-foreground">{t("mcpServers.additionalAuth.description")}</p>
      </div>
      {value.map((draft, index) => (
        <ProfileRow
          key={draft.rowKey}
          draft={draft}
          drafts={value}
          onChange={(next) => onChange(value.map((existing, at) => (at === index ? next : existing)))}
          onRemove={() => onChange(value.filter((_, at) => at !== index))}
        />
      ))}
      <Button
        variant="outline"
        className="w-full border-dashed"
        onClick={() => onChange([...value, newProfileDraft()])}
      >
        <Plus />
        {t("mcpServers.additionalAuth.add")}
      </Button>
    </div>
  );
};

export default AdditionalAuthProfilesSection;
