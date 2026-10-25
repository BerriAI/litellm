import { useLanguage } from "@/contexts/LanguageContext";
import { SquareArrowOutUpRight } from "lucide-react";
import Link from "next/link";

import { userRecordHref } from "../usageUrlState";

export default function UserRecordLink({ userId }: { userId: string }) {
  const { t } = useLanguage();
  return (
    <Link
      href={userRecordHref(userId)}
      className="inline-flex shrink-0 items-center gap-1 text-sm text-primary underline underline-offset-2"
    >
      {t("View user record")}
      <SquareArrowOutUpRight className="size-3.5" />
    </Link>
  );
}
