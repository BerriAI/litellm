"use client";

import EvalsCatalog from "@/components/evals/EvalsCatalog";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const EvalsPage = () => {
  const { accessToken } = useAuthorized();

  return <EvalsCatalog accessToken={accessToken} />;
};

export default EvalsPage;
