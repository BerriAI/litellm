import { Suspense } from "react";

import PublicAuthForm from "@/components/public-relay/PublicAuthForm";

export default function ActivatePage() {
  return (
    <Suspense>
      <PublicAuthForm mode="activate" />
    </Suspense>
  );
}
