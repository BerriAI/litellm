import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createRoot } from 'react-dom/client';
import 'react-resizable/css/styles.css';

import { PlaygroundApp } from './playground-app';
import './style.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: Number.POSITIVE_INFINITY },
  },
});

const root = document.getElementById('root');
if (!root) {
  throw new Error('Missing React root');
}

createRoot(root).render(
  <QueryClientProvider client={queryClient}>
    <PlaygroundApp />
  </QueryClientProvider>,
);
