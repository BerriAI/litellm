import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/outline";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { useLanguage } from "@/contexts/LanguageContext";
import { toast } from "@/lib/toast";
import type { Translate } from "@/i18n/language";

export interface LocalMessage {
  message: string;
  values?: Parameters<Translate>[1];
}

export interface UserData {
  user_email: string;
  user_role: string;
  teams?: string | string[];
  metadata?: string;
  max_budget?: string | number;
  budget_duration?: string;
  models?: string | string[];
  status?: string;
  error?: string;
  validationErrors?: LocalMessage[];
  localError?: LocalMessage;
  rowNumber?: number;
  isValid?: boolean;
  key?: string;
  invitation_link?: string;
}

// 行数据保留原始错误供导出，界面仅翻译本地生成的校验模板。
export default function BulkUserStatusCell({ record }: { record: UserData }) {
  const { t, language } = useLanguage();
  if (!record.isValid) {
    return (
      <div>
        <div className="flex items-center">
          <XCircleIcon className="h-5 w-5 text-destructive mr-2" />
          <span className="text-destructive">{t("Invalid")}</span>
        </div>
        {record.error && (
          <span className="text-sm text-destructive ml-7">
            {record.validationErrors
              ? record.validationErrors
                  .map(({ message, values }) => t(message, values))
                  .join(language === "zh-CN" ? "，" : ", ")
              : record.error}
          </span>
        )}
      </div>
    );
  }
  if (!record.status || record.status === "pending") {
    return <span className="text-muted-foreground">{t("Pending")}</span>;
  }
  if (record.status === "success") {
    return (
      <div>
        <div className="flex items-center">
          <CheckCircleIcon className="h-5 w-5 text-success mr-2" />
          <span className="text-success">{t("Success")}</span>
        </div>
        {record.invitation_link && (
          <div className="mt-1">
            <div className="flex items-center">
              <span className="text-xs text-muted-foreground truncate max-w-[150px]">{record.invitation_link}</span>
              <CopyToClipboard text={record.invitation_link} onCopy={() => toast.success(t("Invitation link copied!"))}>
                <button className="ml-1 text-info text-xs hover:text-info/80">{t("Copy")}</button>
              </CopyToClipboard>
            </div>
          </div>
        )}
      </div>
    );
  }
  return (
    <div>
      <div className="flex items-center">
        <XCircleIcon className="h-5 w-5 text-destructive mr-2" />
        <span className="text-destructive">{t("Failed")}</span>
      </div>
      {record.error && (
        <span className="text-sm text-destructive ml-7">
          {record.localError ? t(record.localError.message, record.localError.values) : JSON.stringify(record.error)}
        </span>
      )}
    </div>
  );
}
