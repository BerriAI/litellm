import { Suspense } from "react";

import PublicAuthForm from "@/components/public-relay/PublicAuthForm";

export default function PasswordResetPage() {
  return (
    <Suspense>
      <PublicAuthForm mode="reset" />
    </Suspense>
  );
}
