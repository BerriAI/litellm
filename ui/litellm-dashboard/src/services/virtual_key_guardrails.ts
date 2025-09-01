import { Guardrail } from '../types/guardrail';

interface VirtualKeyGuardrailsResponse {
  virtual_key_id: string;
  guardrails: Guardrail[];
}

export async function fetchGuardrailsForVirtualKey(virtualKeyId: string): Promise<VirtualKeyGuardrailsResponse> {
  const response = await fetch(`/api/guardrails/virtual_key/${virtualKeyId}`);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to fetch guardrails: ${errorText}`);
  }
  
  return await response.json();
}

export async function associateGuardrailWithVirtualKey(virtualKeyId: string, guardrailId: string): Promise<{ message: string }> {
  const response = await fetch('/api/guardrails/virtual_key/associate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      virtual_key_id: virtualKeyId,
      guardrail_id: guardrailId,
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to associate guardrail: ${errorText}`);
  }
  
  return await response.json();
}

export async function disassociateGuardrailFromVirtualKey(virtualKeyId: string, guardrailId: string): Promise<{ message: string }> {
  const response = await fetch('/api/guardrails/virtual_key/disassociate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      virtual_key_id: virtualKeyId,
      guardrail_id: guardrailId,
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to disassociate guardrail: ${errorText}`);
  }
  
  return await response.json();
}
