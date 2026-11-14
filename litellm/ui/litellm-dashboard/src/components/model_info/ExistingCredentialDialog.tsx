import { useTranslation } from "react-i18next";
import type { CredentialItem } from "@/components/networking";
import { CredentialIdentity } from "@/components/shared/CredentialIdentity";
import { credentialLabelsByName } from "@/components/shared/credentialOptions";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export function ExistingCredentialDialog({
  open,
  onClose,
  credentialName,
  credentials,
}: {
  open: boolean;
  onClose: () => void;
  credentialName: string;
  credentials: CredentialItem[];
}) {
  const { t } = useTranslation();
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("modelInfoView.existingCredentialDialog.title")}</DialogTitle>
        </DialogHeader>
        <CredentialIdentity
          className="text-sm"
          credentialName={credentialName}
          label={credentialLabelsByName(credentials).get(credentialName)}
        />
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("modelInfoView.existingCredentialDialog.cancel")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
