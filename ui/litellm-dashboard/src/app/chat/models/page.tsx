"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import ModelHubTable from "@/components/AIHub/ModelHubTable";
import PublicModelHub from "@/components/public_model_hub";
import { useChatShell } from "@/contexts/ChatShellContext";
import { isAdminRole } from "@/utils/roles";

export default function ChatModelsPage() {
  const { t } = useLanguage();
  const { accessToken, userRole, premiumUser } = useChatShell();

  return (
    <div className="moyai-models-page">
      <div className="moyai-models-content">
        <header className="moyai-models-header">
          <p className="text-xs font-medium text-muted-foreground">{t("Model Hub")}</p>
          <h1>{t("Models")}</h1>
          <p className="text-sm text-muted-foreground">
            {t("Find the right model. Explore providers, compare pricing, and see what each model can do.")}
          </p>
        </header>
        {isAdminRole(userRole) ? (
          <ModelHubTable
            accessToken={accessToken}
            publicPage={false}
            premiumUser={premiumUser}
            userRole={userRole}
            canOpenPlayground={false}
            canAddModels={false}
            variant="chat"
          />
        ) : (
          <PublicModelHub accessToken={accessToken} isEmbedded variant="chat" />
        )}
      </div>
    </div>
  );
}
