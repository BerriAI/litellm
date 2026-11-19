import { useQuery } from "@tanstack/react-query";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { loadApps } from "./appApi";

interface AppAccessSelectorProps {
  id?: string;
  "aria-describedby"?: string;
  value?: string[];
  onChange: (appIds: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
}

export default function AppAccessSelector({
  id,
  "aria-describedby": describedBy,
  value,
  onChange,
  placeholder,
  disabled = false,
}: AppAccessSelectorProps) {
  const apps = useQuery({ queryKey: ["apps", "access-selector"], queryFn: () => loadApps(), enabled: !disabled });
  const options = (apps.data ?? []).map((app) => ({ label: `${app.alias} (${app.app_id})`, value: app.app_id }));
  return (
    <div className="min-w-0">
      <MultiSelect
        id={id}
        aria-describedby={describedBy}
        placeholder={placeholder}
        onValueChange={onChange}
        value={value}
        loading={apps.isLoading}
        disabled={disabled}
        options={options}
      />
    </div>
  );
}
