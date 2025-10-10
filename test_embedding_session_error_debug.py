"""
Test script to reproduce and debug the ClientSession JSON serialization error
"""
import asyncio
import aiohttp
import litellm
import traceback

# Enable debug mode
litellm._turn_on_debug()

async def test_embedding_with_shared_session():
    """Test that reproduces the ClientSession serialization error"""
    
    async with aiohttp.ClientSession() as session:
        print(f"Created ClientSession: {session}")
        
        try:
            response = await litellm.aembedding(
                model="openai/text-embedding-ada-002",
                input=["test input"],
                api_key="invalid_key",
                api_base="https://httpstat.us/500",  # This will return 500 error
                shared_session=session  # This is the problematic parameter
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"\n❌ Error occurred: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            
            # Check if this is the ClientSession serialization error
            if "ClientSession" in str(e) and "JSON serializable" in str(e):
                print("\n✅ Successfully reproduced the ClientSession serialization error!")
                return True
    
    return False

if __name__ == "__main__":
    asyncio.run(test_embedding_with_shared_session())

