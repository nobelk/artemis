"""Simple test to verify the application can initialize without errors."""
from src.agent.assistant import create_assistant

def test_initialization():
    """Test that the assistant can be created successfully."""
    print("Testing assistant initialization...")
    bot = create_assistant()
    print("✓ Assistant created successfully")
    print(f"✓ Assistant type: {type(bot).__name__}")
    print(f"✓ Assistant has {len(bot.function_list) if hasattr(bot, 'function_list') else 0} tools")
    print("\nAll initialization checks passed!")
    return True

if __name__ == "__main__":
    test_initialization()