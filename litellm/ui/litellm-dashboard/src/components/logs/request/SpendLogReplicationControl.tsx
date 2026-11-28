import { useTranslation } from "react-i18next";

import { Switch } from "@/components/ui/switch";
import { $api } from "@/lib/http/api";
import { toast } from "@/lib/toast";

export default function SpendLogReplicationControl({ readOnly }: { readOnly: boolean }) {
  const { t } = useTranslation();
  const query = $api.useQuery("get", "/spend/logs/replication");
  const update = $api.useMutation("patch", "/spend/logs/replication", {
    onSuccess: () => query.refetch(),
    onError: (error) => toast.fromError(error),
  });
  const settings = query.data;

  if (!settings) return null;

  return (
    <div className="mb-4 flex items-start gap-3 rounded-lg border p-3">
      <Switch
        aria-label={t("logs.replication.label")}
        aria-describedby="request-log-capture-description"
        checked={settings.enabled}
        disabled={readOnly || update.isPending}
        onCheckedChange={(enabled) => update.mutate({ body: { enabled } })}
      />
      <div className="text-sm">
        <div className="font-medium">{t("logs.replication.label")}</div>
        <div id="request-log-capture-description" className="text-muted-foreground">
          {t("logs.replication.description", { region: settings.region })}
        </div>
      </div>
    </div>
  );
}
