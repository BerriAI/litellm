import { tempBudgetExpiryMs } from "@/components/shared/InheritedBudgetHint";
import type { UserInfoV2Response } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { utcIsoToPickerValue } from "@/utils/ptuDatetime";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

const MAX_TIMER_DELAY_MS = 2 ** 31 - 1;

export function useUserTemporaryBudget(user: UserInfoV2Response | null) {
  const expiry = tempBudgetExpiryMs(user);
  const expiryMs = expiry !== null && Number.isFinite(expiry) ? expiry : null;
  const [observedTime, setObservedTime] = useState(() => Date.now());
  const expired = expiryMs !== null && expiryMs <= observedTime;

  useEffect(() => {
    if (expiryMs === null || expired) return;
    const observeExpiry = () => {
      const now = Date.now();
      if (expiryMs <= now) {
        setObservedTime(now);
      } else {
        timer = setTimeout(observeExpiry, Math.min(expiryMs - now, MAX_TIMER_DELAY_MS));
      }
    };
    let timer = setTimeout(observeExpiry, Math.min(Math.max(expiryMs - Date.now(), 0), MAX_TIMER_DELAY_MS));
    return () => clearTimeout(timer);
  }, [expiryMs, expired]);

  const effectiveBudget =
    expired && user?.max_budget != null
      ? user.max_budget + (user.rollover_credit ?? 0)
      : user?.effective_max_budget ?? user?.max_budget ?? null;
  return { expiryMs, expired, effectiveBudget };
}

export function UserTemporaryBudget({
  user,
  budget,
}: {
  user: UserInfoV2Response;
  budget: ReturnType<typeof useUserTemporaryBudget>;
}) {
  const { t } = useTranslation();
  if (user.temp_budget_increase == null || budget.expiryMs === null) return null;
  return (
    <>
      <div>
        <p className="font-medium">{t("users.temporaryBudget.increase")}</p>
        <p>
          +${formatNumberWithCommas(user.temp_budget_increase, 4)} ·{" "}
          {t(budget.expired ? "users.temporaryBudget.expired" : "users.temporaryBudget.active")}
        </p>
      </div>
      <div>
        <p className="font-medium">{t("users.temporaryBudget.expiry")}</p>
        <p>{utcIsoToPickerValue(user.temp_budget_expiry)?.format("YYYY-MM-DD HH:mm:ss")}</p>
      </div>
      <div>
        <p className="font-medium">{t("users.temporaryBudget.effective")}</p>
        <p>
          {budget.effectiveBudget === null
            ? t("users.userInfo.unlimited")
            : `$${formatNumberWithCommas(budget.effectiveBudget, 4)}`}
        </p>
      </div>
    </>
  );
}
