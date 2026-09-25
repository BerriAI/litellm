import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { components } from "@/lib/http/schema";
import { apiClient } from "@/components/networking";
import AccessGroupSelector from "@/components/common_components/AccessGroupSelector";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

type Source = components["schemas"]["SCIMSourceResponse"];
type Mapping = components["schemas"]["SCIMGroupMapping"];

export const SCIMAgentProvisioning = ({ accessToken }: { accessToken: string | null }) => {
  const [editing, setEditing] = useState<Source | null>(null);
  const [name, setName] = useState("");
  const [tenant, setTenant] = useState("");
  const [token, setToken] = useState("");
  const [enabled, setEnabled] = useState(true);
  const [mappings, setMappings] = useState<Mapping[]>([]);
  const [saving, setSaving] = useState(false);
  const sources = useQuery({
    queryKey: ["scim-agent-sources"],
    queryFn: () => apiClient.get<Source[]>("/scim/v2/sources", { accessToken: accessToken ?? "" }),
    enabled: Boolean(accessToken),
  });

  const edit = (source: Source | null) => {
    setEditing(source);
    setName(source?.display_name ?? "");
    setTenant(source?.tenant_id ?? "");
    setEnabled(source?.enabled ?? true);
    setMappings(source?.group_mappings ?? []);
    setToken("");
  };

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!accessToken) return;
    setSaving(true);
    const body = { display_name: name, tenant_id: tenant, enabled, group_mappings: mappings };
    try {
      const saved = editing
        ? await apiClient.put<Source>(`/scim/v2/sources/${encodeURIComponent(editing.source_id)}`, {
            accessToken,
            body,
          })
        : await apiClient.post<Source>("/scim/v2/sources", {
            accessToken,
            body: { ...body, provisioning_token: token },
          });
      edit(saved);
      await sources.refetch();
      toast.success("Agent provisioning configuration saved");
    } catch (error) {
      toast.fromError(error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section aria-label="Entra agent provisioning" className="mt-8 space-y-4 rounded-lg border p-4">
      <h3 className="text-lg font-medium">Entra agent provisioning</h3>
      <p className="text-sm text-muted-foreground">
        Sync Entra agent-user accounts into Agents. New agents start disabled until you configure their permissions and
        enable them. Application service principals can be registered directly in Agents.
      </p>
      {sources.isError && <p role="alert">Could not load provisioning sources</p>}
      <div className="flex flex-wrap gap-2">
        {sources.data?.map((source) => (
          <Button key={source.source_id} type="button" variant="outline" onClick={() => edit(source)}>
            {source.display_name}
          </Button>
        ))}
        <Button type="button" variant="outline" onClick={() => edit(null)}>
          New source
        </Button>
      </div>
      <form onSubmit={save} className="space-y-4">
        <label className="block text-sm">
          Source name
          <Input required value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label className="block text-sm">
          Entra tenant ID
          <Input
            required
            disabled={Boolean(editing)}
            value={tenant}
            onChange={(event) => setTenant(event.target.value)}
          />
        </label>
        {!editing && (
          <label className="block text-sm">
            Dedicated SCIM token
            <Input
              required
              type="password"
              autoComplete="off"
              value={token}
              onChange={(event) => setToken(event.target.value)}
            />
            <span className="text-muted-foreground">Use a token created above, restricted to SCIM routes</span>
          </label>
        )}
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />
          Enable this provisioning source
        </label>
        <p className="text-sm text-muted-foreground">
          Map Entra group object IDs to existing access groups. Provisioned agents need a mapped group as well as their
          own resource permissions.
        </p>
        {mappings.map((mapping, index) => (
          <div key={index} className="space-y-2 rounded border p-3">
            <label className="block text-sm">
              Entra group object ID
              <Input
                required
                value={mapping.external_group_id}
                onChange={(event) =>
                  setMappings((current) =>
                    current.map((item, position) =>
                      position === index ? { ...item, external_group_id: event.target.value } : item,
                    ),
                  )
                }
              />
            </label>
            <AccessGroupSelector
              value={mapping.access_group_ids}
              onChange={(ids) =>
                setMappings((current) =>
                  current.map((item, position) => (position === index ? { ...item, access_group_ids: ids } : item)),
                )
              }
              showLabel
            />
            <Button
              type="button"
              variant="ghost"
              onClick={() => setMappings((current) => current.filter((_, position) => position !== index))}
            >
              Remove mapping
            </Button>
          </div>
        ))}
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => setMappings((current) => [...current, { external_group_id: "", access_group_ids: [] }])}
          >
            Add group mapping
          </Button>
          <Button type="submit" disabled={saving}>
            {saving ? "Saving" : "Save provisioning source"}
          </Button>
        </div>
      </form>
      <p className="text-sm text-muted-foreground">
        In Entra provisioning, map objectId to externalId and identityParentId to the LiteLLM agent-user extension. Use
        the SCIM URL above and sync only assigned users and groups.
      </p>
      <code className="block break-all text-xs">
        urn:ietf:params:scim:schemas:extension:litellmAgent:2.0:User:identityParentId
      </code>
    </section>
  );
};
