import { Guardrail } from '../types/guardrail';

interface GuardrailsResponse {
  guardrails: Guardrail[];
}

export async function fetchGuardrails(): Promise<GuardrailsResponse> {
  const response = await fetch('/api/guardrails');
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to fetch guardrails: ${errorText}`);
  }
  
  return await response.json();
}

export async function fetchGuardrail(guardrailId: string): Promise<Guardrail> {
  const response = await fetch(`/api/guardrails/${guardrailId}`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to fetch guardrail: ${errorText}`);
  }
  
  return await response.json();
}
