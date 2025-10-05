import { useEffect, useState } from "react";
import { fetchGuardrails, fetchPrompts } from "@/app/(console)/virtual-keys/components/CreateKeyModal/networking";
import useAuthorized from "@/app/(console)/hooks/useAuthorized";

export const useGuardrailsAndPrompts = () => {
  const [guardrails, setGuardrails] = useState<string[]>([]);
  const [prompts, setPrompts] = useState<string[]>([]);
  const { accessToken } = useAuthorized();

  useEffect(() => {
    if (!accessToken) {
      setGuardrails([]);
      setPrompts([]);
      return;
    }
    (async () => {
      const [g, p] = await Promise.all([fetchGuardrails(accessToken), fetchPrompts(accessToken)]);
      setGuardrails(g || []);
      setPrompts(p || []);
    })();
  }, [accessToken]);

  return { guardrails, prompts };
};
