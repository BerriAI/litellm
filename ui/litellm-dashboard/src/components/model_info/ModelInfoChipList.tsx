import { useLanguage } from "@/contexts/LanguageContext";
import { Badge } from "@/components/ui/badge";

// 仅翻译本地空状态，访问组、标签等业务名称按原值展示。
export default function ModelInfoChipList({ values, emptyLabel }: { values: unknown; emptyLabel: string }) {
  const { t } = useLanguage();
  if (!values) {
    return <>{t("Not Set")}</>;
  }
  if (!Array.isArray(values)) {
    return <>{String(values)}</>;
  }
  if (values.length === 0) {
    return <>{t(emptyLabel)}</>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {values.map((entry: string, index: number) => (
        <Badge key={index} variant="secondary">
          {entry}
        </Badge>
      ))}
    </div>
  );
}
