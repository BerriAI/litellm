import { SquareArrowOutUpRight } from "lucide-react";
import Link from "next/link";
import { useTranslation } from "react-i18next";

import { userRecordHref } from "../usageUrlState";

export default function UserRecordLink({ userId }: { userId: string }) {
  const { t } = useTranslation();
  const href = userRecordHref(userId);
  if (!href) return null;

  return (
    <Link
      href={href}
      className="inline-flex shrink-0 items-center gap-1 text-sm text-primary underline underline-offset-2"
    >
      {t("usage.userRecordLink.view")}
      <SquareArrowOutUpRight className="size-3.5" />
    </Link>
  );
}
