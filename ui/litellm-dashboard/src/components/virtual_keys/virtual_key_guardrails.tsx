import React, { useEffect, useState } from 'react';
import { Badge, Button, Card, Flex, Heading, Text, Box, Spinner } from '@radix-ui/themes';
import { fetchGuardrailsForVirtualKey, associateGuardrailWithVirtualKey, disassociateGuardrailFromVirtualKey } from '../../services/virtual_key_guardrails';
import { Guardrail } from '../../types/guardrail';
import { toast } from 'react-hot-toast';

interface VirtualKeyGuardrailsProps {
  virtualKeyId: string;
  refreshTrigger?: number;
}

export const VirtualKeyGuardrails: React.FC<VirtualKeyGuardrailsProps> = ({ 
  virtualKeyId,
  refreshTrigger = 0
}) => {
  const [guardrails, setGuardrails] = useState<Guardrail[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadGuardrails = async () => {
      if (!virtualKeyId) return;
      
      setLoading(true);
      try {
        const data = await fetchGuardrailsForVirtualKey(virtualKeyId);
        setGuardrails(data.guardrails);
        setError(null);
      } catch (err) {
        console.error('Error loading guardrails for virtual key:', err);
        setError('Failed to load guardrails');
        setGuardrails([]);
      } finally {
        setLoading(false);
      }
    };

    loadGuardrails();
  }, [virtualKeyId, refreshTrigger]);

  const handleRemoveGuardrail = async (guardrailId: string) => {
    try {
      await disassociateGuardrailFromVirtualKey(virtualKeyId, guardrailId);
      setGuardrails(guardrails.filter(g => g.guardrail_id !== guardrailId));
      toast.success('Guardrail removed from virtual key');
    } catch (err) {
      console.error('Error removing guardrail:', err);
      toast.error('Failed to remove guardrail');
    }
  };

  if (loading) {
    return (
      <Box py="4">
        <Flex align="center" justify="center">
          <Spinner size="large" />
        </Flex>
      </Box>
    );
  }

  if (error) {
    return (
      <Box py="4">
        <Text color="red">{error}</Text>
      </Box>
    );
  }

  return (
    <Card>
      <Heading size="3" mb="2">Attached Guardrails</Heading>
      {guardrails.length === 0 ? (
        <Text color="gray">No guardrails attached to this virtual key</Text>
      ) : (
        <Flex direction="column" gap="2">
          {guardrails.map((guardrail) => (
            <Flex key={guardrail.guardrail_id} justify="between" align="center" p="2" style={{ borderBottom: '1px solid var(--gray-4)' }}>
              <Flex direction="column" gap="1">
                <Text weight="bold">{guardrail.guardrail_name || 'Unnamed Guardrail'}</Text>
                <Text size="1" color="gray">Type: {guardrail.litellm_params?.guardrail}</Text>
                <Text size="1" color="gray">Mode: {guardrail.litellm_params?.mode}</Text>
              </Flex>
              <Button color="red" variant="soft" onClick={() => handleRemoveGuardrail(guardrail.guardrail_id)}>
                Remove
              </Button>
            </Flex>
          ))}
        </Flex>
      )}
    </Card>
  );
};
