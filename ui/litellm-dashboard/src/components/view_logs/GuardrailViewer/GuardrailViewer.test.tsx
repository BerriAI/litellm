import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, screen } from "../../../../tests/test-utils"
import {
  makeBedrockResponse, makeEntity,
  makeGuardrailInformation,
} from "@/components/view_logs/GuardrailViewer/__tests__/fixtures"
import GuardrailViewer from "@/components/view_logs/GuardrailViewer/GuardrailViewer"

// We will mock child components selectively for some tests to assert prop passthrough,
// but also run an integration-style render without mocks.
const PresidioPath = '@/components/view_logs/GuardrailViewer/PresidioDetectedEntities';
const BedrockPath = '@/components/view_logs/GuardrailViewer/BedrockGuardrailDetails';

describe('GuardrailViewer', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('shows header, status pill color, duration rounding, and time labels', () => {
    const data = makeGuardrailInformation({ duration: 1.23456, guardrail_status: 'success' });
    renderWithProviders(<GuardrailViewer data={data} />);

    expect(screen.getByText('Guardrail Information')).toBeInTheDocument();
    // header status pill (success => green)
    const statusBadges = screen.getAllByText('success');
    // there are two status locations: header chip and grid "Status"
    expect(statusBadges.length).toBeGreaterThanOrEqual(1);
    // Quick class assertion for at least one of them
    expect(statusBadges[0].className).toMatch(/bg-green-100/);

    // duration displays with 4 decimals
    expect(screen.getByText(/1\.2346s/)).toBeInTheDocument();

    // time labels exist
    expect(screen.getByText('Start Time:')).toBeInTheDocument();
    expect(screen.getByText('End Time:')).toBeInTheDocument();
  });

  it('calculates and displays masked entity totals with pluralization', () => {
    const data = makeGuardrailInformation({
      masked_entity_count: { EMAIL_ADDRESS: 2, PHONE_NUMBER: 1 },
    });
    renderWithProviders(<GuardrailViewer data={data} />);

    expect(screen.getByText('3 masked entities')).toBeInTheDocument();
    // summary chips for each entry
    expect(screen.getByText('EMAIL_ADDRESS: 2')).toBeInTheDocument();
    expect(screen.getByText('PHONE_NUMBER: 1')).toBeInTheDocument();
  });

  it('hides masked badge & summary when count is zero/empty', () => {
    const data = makeGuardrailInformation({ masked_entity_count: {} });
    renderWithProviders(<GuardrailViewer data={data} />);

    expect(screen.queryByText(/masked entity/)).not.toBeInTheDocument();
    expect(screen.queryByText('Masked Entity Summary')).not.toBeInTheDocument();
  });

  it('toggles main section open/closed and chevron rotation class', async () => {
    const user = userEvent.setup();
    const data = makeGuardrailInformation();
    renderWithProviders(<GuardrailViewer data={data} />);

    const header = screen.getByText('Guardrail Information').closest('div')!;
    // Initially expanded
    expect(screen.getByText('Click to collapse')).toBeInTheDocument();
    // Click to collapse
    await user.click(header);
    expect(screen.getByText('Click to expand')).toBeInTheDocument();
    // Details gone
    expect(screen.queryByText('Masked Entity Summary')).not.toBeInTheDocument();

    // Click to expand again
    await user.click(header);
    expect(screen.getByText('Click to collapse')).toBeInTheDocument();
  });

  it('defaults to presidio provider when guardrail_provider is undefined', async () => {
    vi.doMock(PresidioPath, () => ({
      __esModule: true,
      default: ({ entities }: any) => <div data-testid="presidio-mock">presidio {entities?.length}</div>,
    }));
    const { default: Component } = await import('@/components/view_logs/GuardrailViewer/GuardrailViewer');

    const data = makeGuardrailInformation({
      guardrail_provider: undefined,
      guardrail_response: [makeEntity(), makeEntity()],
    });
    renderWithProviders(<Component data={data} />);

    expect(screen.getByTestId('presidio-mock')).toHaveTextContent('presidio 2');
  });

  it('renders PresidioDetectedEntities when provider="presidio" and response has entities', async () => {
    vi.doMock(PresidioPath, () => ({
      __esModule: true,
      default: ({ entities }: any) => <div data-testid="presidio-mock">count:{entities?.length}</div>,
    }));
    const { default: Component } = await import('@/components/view_logs/GuardrailViewer/GuardrailViewer');

    const data = makeGuardrailInformation({
      guardrail_provider: 'presidio',
      guardrail_response: [makeEntity()],
    });
    renderWithProviders(<Component data={data} />);
    expect(screen.getByTestId('presidio-mock')).toHaveTextContent('count:1');
  });

  it('renders BedrockGuardrailDetails when provider="bedrock"', async () => {
    vi.doMock(BedrockPath, () => ({
      __esModule: true,
      default: ({ response }: any) => (
        <div data-testid="bedrock-mock">{response?.action ?? 'no-action'}</div>
      ),
    }));
    const { default: Component } = await import('@/components/view_logs/GuardrailViewer/GuardrailViewer');

    const data = makeGuardrailInformation({
      guardrail_provider: 'bedrock',
      guardrail_response: makeBedrockResponse({ action: 'GUARDRAIL_INTERVENED' }),
    });
    renderWithProviders(<Component data={data} />);
    expect(screen.getByTestId('bedrock-mock')).toHaveTextContent('GUARDRAIL_INTERVENED');
  });

  it('unknown provider renders neither Presidio nor Bedrock details', () => {
    const data = makeGuardrailInformation({
      guardrail_provider: 'unknown',
    });
    renderWithProviders(<GuardrailViewer data={data} />);
    // Summary still present
    expect(screen.getByText('Guardrail Information')).toBeInTheDocument();
    // No provider sections
    expect(screen.queryByText(/Detected Entities/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Raw Bedrock Guardrail Response/)).not.toBeInTheDocument();
  });

  it('integration: renders with real Bedrock details without mocks', () => {
    const data = makeGuardrailInformation({
      guardrail_provider: 'bedrock',
      guardrail_response: makeBedrockResponse({
        action: 'NONE',
        outputs: [{ text: 'ok' }],
      }),
    });
    renderWithProviders(<GuardrailViewer data={data} />);

    // Bedrock summary bits
    expect(screen.getByText('Outputs')).toBeInTheDocument();
    expect(screen.getByText('ok')).toBeInTheDocument();
  });
});
