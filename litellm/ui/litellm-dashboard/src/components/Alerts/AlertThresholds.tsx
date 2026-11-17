import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { components } from "@/lib/http/schema";

type Thresholds = components["schemas"]["PerformanceAlertThresholds"];
const rules = ["spend", "traffic", "latency", "reliability"] as const;

export default function AlertThresholds({
  thresholds,
  enterprise,
  readOnly,
  pending,
  save,
}: {
  thresholds: Thresholds;
  enterprise: boolean;
  readOnly: boolean;
  pending: boolean;
  save: (thresholds: Thresholds) => Promise<unknown>;
}) {
  const { t, i18n } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const disabled = !enterprise || readOnly || pending;
  const format = new Intl.NumberFormat(i18n.language, { maximumSignificantDigits: 15 });
  const minimum = (rule: (typeof rules)[number]) =>
    Number((thresholds[rule].minimum_increase * (rule === "reliability" ? 100 : 1)).toPrecision(15));
  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const threshold = (rule: (typeof rules)[number]) => ({
      multiplier: Number(values.get(`${rule}-multiplier`)),
      minimum_increase: Number(values.get(`${rule}-minimum`)) / (rule === "reliability" ? 100 : 1),
    });
    setSaved(false);
    try {
      const updated = {
        spend: threshold("spend"),
        traffic: threshold("traffic"),
        latency: threshold("latency"),
        reliability: threshold("reliability"),
      };
      await save(updated);
      setEditing(false);
      setSaved(true);
    } catch {
      setSaved(false);
    }
  }
  return (
    <form onSubmit={(event) => void submit(event)} onChange={() => setSaved(false)} className="space-y-4 py-4">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">{t("alerts.thresholds.title")}</h3>
        <p className="text-sm text-muted-foreground">{t("alerts.thresholds.description")}</p>
        {!enterprise && <p className="text-sm text-muted-foreground">{t("alerts.thresholds.enterpriseOnly")}</p>}
      </div>
      {!editing && (
        <dl className="space-y-3 text-sm">
          {rules.map((rule) => (
            <div key={rule} className="space-y-1">
              <dt className="font-medium">{t(`alerts.thresholds.rules.${rule}`)}</dt>
              <dd className="tabular-nums text-muted-foreground">
                {t(`alerts.thresholds.summary.${rule}`, {
                  multiplier: format.format(thresholds[rule].multiplier),
                  minimum: format.format(minimum(rule)),
                })}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {editing &&
        rules.map((rule) => (
          <fieldset key={rule} disabled={disabled} className="grid grid-cols-2 gap-3">
            <legend className="mb-2 text-sm font-medium">{t(`alerts.thresholds.rules.${rule}`)}</legend>
            <div className="space-y-1">
              <label htmlFor={`${rule}-multiplier`} className="text-xs text-muted-foreground">
                {t("alerts.thresholds.multiplier")}
              </label>
              <Input
                id={`${rule}-multiplier`}
                name={`${rule}-multiplier`}
                type="number"
                min="1"
                step="any"
                required
                defaultValue={thresholds[rule].multiplier}
                onChange={(event) =>
                  event.currentTarget.setCustomValidity(
                    event.currentTarget.valueAsNumber <= 1 ? t("alerts.thresholds.multiplierInvalid") : "",
                  )
                }
              />
            </div>
            <div className="space-y-1">
              <label htmlFor={`${rule}-minimum`} className="text-xs text-muted-foreground">
                {t(`alerts.thresholds.minimum.${rule}`)}
              </label>
              <Input
                id={`${rule}-minimum`}
                name={`${rule}-minimum`}
                type="number"
                min="0"
                step="any"
                required
                max={rule === "reliability" ? 100 : undefined}
                defaultValue={minimum(rule)}
              />
            </div>
          </fieldset>
        ))}
      <p className="text-xs text-muted-foreground">{t("alerts.thresholds.safeguards")}</p>
      {enterprise && !readOnly && (
        <div className="flex items-center gap-3">
          {editing ? (
            <>
              <Button type="submit" size="sm" disabled={pending}>
                {t(pending ? "alerts.thresholds.saving" : "alerts.thresholds.save")}
              </Button>
              <Button type="button" variant="ghost" size="sm" disabled={pending} onClick={() => setEditing(false)}>
                {t("alerts.thresholds.cancel")}
              </Button>
            </>
          ) : (
            <Button type="button" variant="outline" size="sm" onClick={() => setEditing(true)}>
              {t("alerts.thresholds.edit")}
            </Button>
          )}
          {saved && (
            <span role="status" className="text-sm text-muted-foreground">
              {t("alerts.thresholds.saved")}
            </span>
          )}
        </div>
      )}
    </form>
  );
}
