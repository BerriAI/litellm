import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TruncatedValue } from "./TruncatedValue";

interface CustomerSectionProps {
  metadata?: Record<string, unknown> | null;
}

export function CustomerSection({ metadata }: CustomerSectionProps) {
  const { t } = useTranslation();
  const rows = (["org", "team", "member"] as const).flatMap((scope) => {
    const id = metadata?.[`user_api_key_customer_${scope}_id`];
    const alias = metadata?.[`user_api_key_customer_${scope}_alias`];
    const customerId = typeof id === "string" && id.trim() ? id : undefined;
    const customerAlias = typeof alias === "string" && alias.trim() ? alias : undefined;
    if (!customerId && !customerAlias) return [];
    return [{ scope, customerId, customerAlias }];
  });

  if (rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("logs.detail.customer.title")}</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3">
          {rows.map(({ scope, customerId, customerAlias }) => (
            <div key={scope} className="grid grid-cols-[8rem_minmax(0,1fr)] items-start gap-3 text-sm">
              <dt className="text-muted-foreground">{t(`logs.detail.customer.${scope}`)}</dt>
              <dd className="min-w-0 space-y-1">
                {customerAlias && customerAlias !== customerId && (
                  <div>
                    <TruncatedValue value={customerAlias} maxWidth={360} />
                  </div>
                )}
                {customerId && (
                  <div>
                    <TruncatedValue value={customerId} maxWidth={360} />
                  </div>
                )}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
