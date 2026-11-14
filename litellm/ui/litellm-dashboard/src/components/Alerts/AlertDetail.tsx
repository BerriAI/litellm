import { ArrowLeft, KeyRound, Network, Server, User, Users } from "lucide-react";
import { useTranslation } from "react-i18next";
import { buttonVariants } from "@/components/ui/button";
import RecommendedReplacement from "@/components/ModelDeprecationAlerts/RecommendedReplacement";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import AlertActions from "./AlertActions";
import AlertChart from "./AlertChart";
import { AlertItem, dateLabel, presentAlert } from "./alertPresentation";

export default function AlertDetail({
  item,
  readOnly,
  onBack,
}: {
  item: AlertItem;
  readOnly: boolean;
  onBack: () => void;
}) {
  const { t } = useTranslation();
  const alert = presentAlert(item);
  const Icon = { key: KeyRound, team: Users, user: User, model: Network, gateway: Server }[alert.resourceType];
  return (
    <div className="min-w-0 space-y-6">
      <button
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        onClick={onBack}
      >
        <ArrowLeft className="size-4" />
        {t("alerts.unified.title")}
      </button>
      <div className="space-y-7">
        <header className="space-y-3">
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:gap-6">
            <h1 className="min-w-0 break-words text-2xl font-semibold leading-snug">{alert.title}</h1>
            <div className="shrink-0 sm:pt-0.5">
              <AlertActions item={item} readOnly={readOnly} />
            </div>
          </div>
          <p className="text-base leading-relaxed text-muted-foreground">{alert.description}</p>
        </header>
        <AlertChart alert={alert} />
        <section className="space-y-2 border-t border-border pt-5">
          <h2 className="text-sm font-medium">{t("alerts.unified.resourceHeading")}</h2>
          <ul className="divide-y divide-border">
            <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-8 gap-y-4 border-b border-border py-4 lg:grid-cols-[minmax(0,1fr)_minmax(280px,1fr)_auto]">
              <div className="order-1 flex min-w-0 items-center gap-2.5 text-sm">
                <Icon strokeWidth={1.75} className="size-[18px] shrink-0 text-muted-foreground" />
                <span className="break-all font-medium">{alert.resourceName}</span>
              </div>
              <dl className="order-3 col-span-2 grid grid-cols-2 gap-x-8 text-xs lg:order-2 lg:col-span-1 lg:gap-x-10">
                {alert.dates.map(([label, date]) => (
                  <div key={label}>
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="mt-1">{dateLabel(date)} UTC</dd>
                  </div>
                ))}
              </dl>
              {alert.href && (
                <a
                  className={buttonVariants({ variant: "outline", size: "sm", className: "order-2 lg:order-3" })}
                  href={alert.href}
                >
                  {alert.linkLabel}
                </a>
              )}
            </li>
          </ul>
        </section>
        {item.source === "deprecation" && (
          <section className="space-y-3">
            <RecommendedReplacement recommendation={item.alert.model} />
            <a
              className={buttonVariants({ variant: "outline", size: "sm" })}
              href={`${migrationHref(item.alert.usageModel)}&migration=true`}
            >
              {t("alerts.unified.migrationDetails")}
            </a>
          </section>
        )}
      </div>
    </div>
  );
}
