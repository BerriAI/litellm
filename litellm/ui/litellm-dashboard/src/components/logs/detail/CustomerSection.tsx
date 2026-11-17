import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import { customerHierarchyHref } from "@/app/(dashboard)/customers/_components/CustomerHierarchyNavigation";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { customerLogIdentity, type CustomerLogRow } from "../customerIdentity";
import { TruncatedValue } from "./TruncatedValue";

interface CustomerSectionProps {
  metadata?: Record<string, unknown> | null;
  endUser?: string;
}

export function CustomerSection({ metadata, endUser }: CustomerSectionProps) {
  const { t } = useTranslation();
  const { rows, summary, recognized } = customerLogIdentity(metadata, endUser);
  if (rows.length === 0) return null;

  const label = (row: CustomerLogRow) => t(`logs.detail.customer.${row.scope === "legacy" ? "title" : row.scope}`);
  const org = rows.find((row) => row.scope === "org")?.id;
  const team = rows.find((row) => row.scope === "team")?.id ?? null;
  const href = (row: CustomerLogRow) => {
    if (!org || !row.id || row.scope === "legacy") return undefined;
    return customerHierarchyHref("manage", {
      org,
      team: row.scope === "org" ? null : team,
      member: row.scope === "member" ? row.id : null,
    });
  };

  return (
    <Card>
      <CardContent className="py-0">
        <Accordion>
          <AccordionItem value="customer">
            <AccordionTrigger className="items-center gap-3 hover:no-underline">
              <span className="shrink-0">{t("logs.detail.customer.title")}</span>
              <span className="grid min-w-0 flex-1 grid-cols-2 gap-3">
                {summary.map((row) => (
                  <span key={row.scope} className="min-w-0 text-xs">
                    <span className="block whitespace-nowrap text-[11px] text-muted-foreground font-normal">
                      {label(row)}
                    </span>
                    <span className="block truncate" title={row.name}>
                      {row.name}
                    </span>
                  </span>
                ))}
              </span>
              <Badge variant={recognized ? "secondary" : "outline"}>
                {t(`logs.detail.customer.${recognized ? "recognized" : "asserted"}`)}
              </Badge>
            </AccordionTrigger>
            <AccordionContent>
              <dl className="space-y-4 pt-2">
                {rows.map((row) => (
                  <div key={row.scope} className="grid grid-cols-[8rem_minmax(0,1fr)_auto] items-start gap-3 text-sm">
                    <dt className="text-muted-foreground">{label(row)}</dt>
                    <dd className="min-w-0 space-y-1">
                      {row.name !== row.id && <div className="break-words font-medium">{row.name}</div>}
                      {row.id && (
                        <div className="text-muted-foreground">
                          <TruncatedValue value={row.id} maxWidth={250} />
                        </div>
                      )}
                    </dd>
                    <dd>
                      {href(row) && (
                        <Button
                          nativeButton={false}
                          role="link"
                          render={<a href={href(row)} />}
                          variant="outline"
                          size="xs"
                          aria-label={t("logs.detail.customer.viewEntity", { entity: label(row) })}
                        >
                          <ExternalLink />
                          {t("logs.detail.customer.view")}
                        </Button>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
