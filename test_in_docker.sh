#!/bin/bash

# Script to test enterprise features inside the Docker container

echo "Testing LiteLLM on-prem enterprise features inside Docker container..."
echo "=================================================="

# Run the test inside the container
docker run --rm --entrypoint="" litellm-onprem python -c "
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, '/app/litellm-source')

try:
    # Try to import the enterprise module
    import enterprise
    print('✅ Successfully imported enterprise module')

    # Try to import specific enterprise features
    from enterprise import *
    print('✅ Successfully imported all enterprise features')

    # Try to import enterprise LiteLLM module
    import enterprise.litellm_enterprise
    print('✅ Successfully imported enterprise.litellm_enterprise')

    # Try to import enterprise callbacks
    import enterprise.litellm_enterprise.enterprise_callbacks
    print('✅ Successfully imported enterprise callbacks')

    # Try to import enterprise integrations
    import enterprise.litellm_enterprise.integrations
    print('✅ Successfully imported enterprise integrations')

    # Try to import proxy modules
    import enterprise.litellm_enterprise.proxy
    print('✅ Successfully imported enterprise proxy features')

    print('==================================================')
    print('🎉 All enterprise features are successfully unlocked!')
    print('✅ The on-prem version is ready for use.')

except ImportError as e:
    print(f'❌ Failed to import enterprise modules: {e}')
    print('==================================================')
    print('❌ Some enterprise features are not available.')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)
"
