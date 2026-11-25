const reference = document.getElementById('reference');

Scalar.createApiReference(reference, {
  url: reference.dataset.openapiUrl,
  servers: [{url: location.origin + (reference.dataset.serverUrl === '/' ? '' : reference.dataset.serverUrl)}],
  hideClientButton: true,
  showDeveloperTools: 'never',
  withDefaultFonts: false,
  telemetry: false,
  agent: {disabled: true},
});
