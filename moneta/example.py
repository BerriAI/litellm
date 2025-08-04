"""
Example usage of the Lago billing integration with LiteLLM.

This module demonstrates how to set up and use the streamlined Lago billing
integration for pre-call entitlement checking and post-call usage reporting.
"""

import os
import asyncio
import litellm
from litellm.moneta import LagoLogger, MonitoringService


def setup_environment():
    """Set up environment variables for testing (replace with your actual values)"""
    # Required configuration
    os.environ["LAGO_API_BASE"] = "https://your-lago-instance.com"
    os.environ["LAGO_API_KEY"] = "your_lago_api_key"
    os.environ["LAGO_PUBLISHER_ID"] = "your_publisher_id"
    
    # Optional configuration
    os.environ["LAGO_TIMEOUT"] = "5"
    os.environ["LAGO_FALLBACK_ALLOW"] = "true"


def basic_setup_example():
    """Example of basic setup"""
    print("=== Basic Lago Integration Setup ===")
    
    try:
        # Create and configure the logger
        lago_logger = LagoLogger()
        
        # Register with LiteLLM
        litellm.callbacks = [lago_logger]
        
        print("✅ Lago integration setup successful!")
        print(f"API Base: {lago_logger.config.api_base}")
        print(f"Publisher ID: {lago_logger.config.publisher_id}")
        print(f"Fallback Allow: {lago_logger.config.fallback_allow}")
        
        return lago_logger
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None


def monitoring_example(lago_logger: LagoLogger):
    """Example of monitoring capabilities"""
    print("\n=== Monitoring Example ===")
    
    # Create monitoring service
    monitor = MonitoringService(lago_logger)
    
    # Get basic statistics
    stats = monitor.get_stats()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Perform health check
    health = monitor.health_check()
    print(f"\nHealth Status: {health['status']}")
    if health['issues']:
        print("Issues:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    # Get configuration summary
    config = monitor.get_config_summary()
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


async def usage_example():
    """Example of actual usage with LiteLLM"""
    print("\n=== Usage Example ===")
    
    try:
        # This would normally be a real LiteLLM completion call
        # The Lago integration will automatically:
        # 1. Check entitlement before the call
        # 2. Report usage after successful completion
        
        print("Making LiteLLM completion call...")
        print("(Lago integration will handle entitlement check and usage reporting)")
        
        # Example call (commented out to avoid actual API calls)
        # response = await litellm.acompletion(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": "Hello!"}],
        #     user="customer_123"  # This becomes the customer_id for billing
        # )
        
        print("✅ Call would complete with automatic billing integration")
        
    except Exception as e:
        print(f"❌ Call failed: {e}")


def error_handling_example():
    """Example of error handling scenarios"""
    print("\n=== Error Handling Examples ===")
    
    # Test with missing configuration
    print("1. Testing with missing configuration...")
    original_key = os.environ.get("LAGO_API_KEY")
    
    try:
        # Temporarily remove API key
        if "LAGO_API_KEY" in os.environ:
            del os.environ["LAGO_API_KEY"]
        
        lago_logger = LagoLogger()
        print("❌ Should have failed with missing configuration")
        
    except ValueError as e:
        print(f"✅ Correctly caught configuration error: {e}")
    
    finally:
        # Restore API key
        if original_key:
            os.environ["LAGO_API_KEY"] = original_key
    
    # Test fallback behavior
    print("\n2. Testing fallback behavior...")
    lago_logger = LagoLogger()
    print(f"Fallback allow setting: {lago_logger.config.fallback_allow}")
    print("When authorization fails, requests will be " + 
          ("allowed" if lago_logger.config.fallback_allow else "blocked"))


def main():
    """Main example function"""
    print("Lago Billing Integration Example")
    print("=" * 40)
    
    # Set up environment (replace with your actual values)
    setup_environment()
    
    # Basic setup
    lago_logger = basic_setup_example()
    
    if lago_logger:
        # Monitoring example
        monitoring_example(lago_logger)
        
        # Usage example
        asyncio.run(usage_example())
        
        # Error handling example
        error_handling_example()
    
    print("\n" + "=" * 40)
    print("Example completed!")


if __name__ == "__main__":
    main()
