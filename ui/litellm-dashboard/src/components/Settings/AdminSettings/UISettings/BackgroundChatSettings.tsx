import { $api } from "@/lib/http/api";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { uiHref } from "@/utils/uiHref";

export default function BackgroundChatSettings() {
  const query = $api.useQuery("get", "/get/background_chat_settings");
  const update = $api.useMutation("patch", "/update/background_chat_settings", {
    onSuccess: () => {
      void query.refetch();
      toast.success("Background chat settings saved.");
    },
    onError: (error) => toast.fromError(error),
  });

  if (query.isPending) return <Skeleton className="h-14 w-full" />;
  if (query.isError || !query.data)
    return (
      <div role="alert" className="text-sm text-destructive">
        Could not load background chat settings.
        <Button variant="ghost" size="sm" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );

  const settings = query.data;
  const readOnly = settings.managed_by_config || !settings.can_save;
  const needsRedis = !settings.redis_configured && !settings.enabled;
  const disabled = update.isPending || readOnly || needsRedis;

  return (
    <div className="flex items-start gap-3">
      <Switch
        checked={settings.enabled}
        disabled={disabled}
        onCheckedChange={(enabled) => update.mutate({ body: { enabled } })}
        aria-label="Continue chats in the background"
        aria-describedby="background-chat-description"
      />
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">Continue chats in the background</p>
        <p id="background-chat-description" className="text-sm text-muted-foreground">
          Keep replies and tool runs going after users close the browser. Return to the same browser to see the result.
        </p>
        {!settings.redis_configured && (
          <p className="text-sm text-muted-foreground">
            Requires Redis.{" "}
            <a className="text-primary underline" href={uiHref("/caching")}>
              Configure it in Caching → Coordination Redis
            </a>
            , then restart the proxy.
          </p>
        )}
        {settings.managed_by_config && (
          <p className="text-sm text-muted-foreground">
            Managed by your config file. Update its background polling setting to change this.
          </p>
        )}
        {!settings.can_save && !settings.managed_by_config && (
          <p className="text-sm text-muted-foreground">Enable database-backed configuration to save this setting.</p>
        )}
        {!readOnly && (
          <p className="text-sm text-muted-foreground">
            Changes apply automatically when the proxy refreshes its configuration.
          </p>
        )}
      </div>
    </div>
  );
}
