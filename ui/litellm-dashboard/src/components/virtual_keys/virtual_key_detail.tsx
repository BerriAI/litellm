import React, { useState, useEffect } from 'react';
import { Card, Flex, Text, Heading, Button, Dialog, TextField, Select, Box, Tabs } from '@radix-ui/themes';
import { toast } from 'react-hot-toast';
import { VirtualKey } from '../../types/virtual_key';
import { fetchVirtualKey, updateVirtualKey } from '../../services/virtual_keys';
import { VirtualKeyGuardrails } from './virtual_key_guardrails';
import { fetchGuardrails } from '../../services/guardrails';
import { Guardrail } from '../../types/guardrail';
import { associateGuardrailWithVirtualKey } from '../../services/virtual_key_guardrails';

interface VirtualKeyDetailProps {
  virtualKeyId: string;
  onClose: () => void;
}

export const VirtualKeyDetail: React.FC<VirtualKeyDetailProps> = ({ virtualKeyId, onClose }) => {
  const [virtualKey, setVirtualKey] = useState<VirtualKey | null>(null);
  const [loading, setLoading] = useState(true);
  const [availableGuardrails, setAvailableGuardrails] = useState<Guardrail[]>([]);
  const [selectedGuardrailId, setSelectedGuardrailId] = useState<string>('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    const loadVirtualKey = async () => {
      try {
        const data = await fetchVirtualKey(virtualKeyId);
        setVirtualKey(data);
      } catch (error) {
        console.error('Error loading virtual key:', error);
        toast.error('Failed to load virtual key details');
      } finally {
        setLoading(false);
      }
    };

    const loadGuardrails = async () => {
      try {
        const data = await fetchGuardrails();
        setAvailableGuardrails(data.guardrails);
      } catch (error) {
        console.error('Error loading guardrails:', error);
      }
    };

    loadVirtualKey();
    loadGuardrails();
  }, [virtualKeyId]);

  const handleAddGuardrail = async () => {
    if (!selectedGuardrailId) {
      toast.error('Please select a guardrail');
      return;
    }

    try {
      await associateGuardrailWithVirtualKey(virtualKeyId, selectedGuardrailId);
      toast.success('Guardrail added to virtual key');
      setSelectedGuardrailId('');
      setRefreshTrigger(prev => prev + 1);
    } catch (error) {
      console.error('Error adding guardrail:', error);
      toast.error('Failed to add guardrail');
    }
  };

  if (loading) {
    return (
      <Dialog.Root open={true} onOpenChange={onClose}>
        <Dialog.Content>
          <Dialog.Title>Virtual Key Details</Dialog.Title>
          <Text>Loading...</Text>
        </Dialog.Content>
      </Dialog.Root>
    );
  }

  if (!virtualKey) {
    return (
      <Dialog.Root open={true} onOpenChange={onClose}>
        <Dialog.Content>
          <Dialog.Title>Error</Dialog.Title>
          <Text>Virtual key not found</Text>
          <Dialog.Close>
            <Button>Close</Button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Root>
    );
  }

  return (
    <Dialog.Root open={true} onOpenChange={onClose}>
      <Dialog.Content style={{ maxWidth: '600px' }}>
        <Dialog.Title>Virtual Key: {virtualKey.key_name || 'Unnamed Key'}</Dialog.Title>
        
        <Tabs.Root defaultValue="details">
          <Tabs.List>
            <Tabs.Trigger value="details">Details</Tabs.Trigger>
            <Tabs.Trigger value="guardrails">Guardrails</Tabs.Trigger>
          </Tabs.List>
          
          <Box py="4">
            <Tabs.Content value="details">
              <Card>
                <Flex direction="column" gap="3">
                  <Flex justify="between">
                    <Text weight="bold">Key ID:</Text>
                    <Text>{virtualKey.key_id}</Text>
                  </Flex>
                  <Flex justify="between">
                    <Text weight="bold">Team ID:</Text>
                    <Text>{virtualKey.team_id || 'N/A'}</Text>
                  </Flex>
                  <Flex justify="between">
                    <Text weight="bold">Models:</Text>
                    <Text>{virtualKey.models?.join(', ') || 'All models'}</Text>
                  </Flex>
                  <Flex justify="between">
                    <Text weight="bold">Spend:</Text>
                    <Text>${virtualKey.spend?.toFixed(2) || '0.00'}</Text>
                  </Flex>
                </Flex>
              </Card>
            </Tabs.Content>
            
            <Tabs.Content value="guardrails">
              <Flex direction="column" gap="3">
                <Card>
                  <Heading size="3" mb="2">Add Guardrail</Heading>
                  <Flex gap="2">
                    <Select.Root value={selectedGuardrailId} onValueChange={setSelectedGuardrailId}>
                      <Select.Trigger placeholder="Select a guardrail" />
                      <Select.Content>
                        {availableGuardrails.map(guardrail => (
                          <Select.Item key={guardrail.guardrail_id} value={guardrail.guardrail_id}>
                            {guardrail.guardrail_name || `Guardrail ${guardrail.guardrail_id.substring(0, 8)}`}
                          </Select.Item>
                        ))}
                      </Select.Content>
                    </Select.Root>
                    <Button onClick={handleAddGuardrail}>Add</Button>
                  </Flex>
                </Card>
                
                <VirtualKeyGuardrails 
                  virtualKeyId={virtualKeyId} 
                  refreshTrigger={refreshTrigger}
                />
              </Flex>
            </Tabs.Content>
          </Box>
        </Tabs.Root>
        
        <Dialog.Close asChild>
          <Button color="gray" variant="soft" mt="4">Close</Button>
        </Dialog.Close>
      </Dialog.Content>
    </Dialog.Root>
  );
};
