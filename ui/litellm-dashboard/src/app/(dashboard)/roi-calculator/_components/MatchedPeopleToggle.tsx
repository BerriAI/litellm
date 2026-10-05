import { useId } from "react";
import { Switch } from "@/components/ui/switch";

export function MatchedPeopleToggle({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const id = useId();
  return (
    <label htmlFor={id} className="flex shrink-0 cursor-pointer items-center gap-2 text-xs text-muted-foreground">
      <Switch id={id} size="sm" checked={checked} onCheckedChange={onChange} />
      Matched people only
    </label>
  );
}
