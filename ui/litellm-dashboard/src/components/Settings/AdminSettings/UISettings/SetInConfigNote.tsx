import { Lock } from "lucide-react";

interface SetInConfigNoteProps {
  settingKey: string;
}

export default function SetInConfigNote({ settingKey }: SetInConfigNoteProps) {
  return (
    <p className="flex items-start gap-1.5 text-xs text-muted-foreground">
      <Lock className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
      <span>
        Set in the proxy config file as <code className="font-mono">general_settings.{settingKey}</code>, which takes
        precedence over this page. Change it in the config file.
      </span>
    </p>
  );
}
