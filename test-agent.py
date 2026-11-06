import os
import json
import shutil
import ollama

# ==========================================
# 1. SET UP THE SOFT SANDBOX
# ==========================================
SANDBOX_DIR = "./sandbox_trial_1"

# Reset the sandbox for a clean test (deletes if it exists, then recreates)
if os.path.exists(SANDBOX_DIR):
    shutil.rmtree(SANDBOX_DIR)
os.makedirs(SANDBOX_DIR)

print(f"📁 Sandbox created at: {os.path.abspath(SANDBOX_DIR)}")

# ==========================================
# 2. DEFINE THE TOOL FOR THE AGENT
# ==========================================
# This is the actual Python function that runs when the AI asks to write a file
def write_file(filename: str, content: str):
    # Force the file to be written INSIDE the sandbox, even if the AI provides a path
    safe_path = os.path.join(SANDBOX_DIR, os.path.basename(filename))
    with open(safe_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote to {filename}"

# This tells the AI what tools it has access to
tools = [{
    'type': 'function',
    'function': {
        'name': 'write_file',
        'description': 'Write text content to a file.',
        'parameters': {
            'type': 'object',
            'properties': {
                'filename': {'type': 'string', 'description': 'The name of the file, e.g., test.txt'},
                'content': {'type': 'string', 'description': 'The text content to write inside the file.'}
            },
            'required': ['filename', 'content']
        }
    }
}]

# ==========================================
# 3. RUN THE AGENT LOOP
# ==========================================
prompt = "Create a file named test.txt containing the text 'Hello World from Qwen'."
print(f"\n🧠 Sending prompt to Qwen2.5 7B: '{prompt}'\n")

messages = [{'role': 'user', 'content': prompt}]

# First API call to the model
response = ollama.chat(
    model='qwen2.5:7b',
    messages=messages,
    tools=tools,
    options={'num_ctx': 4096} # CRITICAL: Limit context to protect your 6GB VRAM
)

# Check if the model decided to call the tool
if response.message.tool_calls:
    for tool in response.message.tool_calls:
        print(f"🔧 Agent decided to use tool: {tool.function.name}")
        print(f"   Arguments provided: {tool.function.arguments}")
        
        # Execute the tool based on the model's arguments
        if tool.function.name == 'write_file':
            args = tool.function.arguments
            result = write_file(args['filename'], args['content'])
            
            # Feed the result back to the model
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': result, 'name': 'write_file'})
            
            # Get the final response from the model
            final_response = ollama.chat(
                model='qwen2.5:7b',
                messages=messages,
                options={'num_ctx': 4096}
            )
            print(f"\n🤖 Agent final answer: {final_response.message.content}")
else:
    print("🤖 Agent responded with text instead of a tool call:")
    print(response.message.content)

# ==========================================
# 4. VERIFY THE RESULT
# ==========================================
print("\n--- VERIFICATION ---")
target_file = os.path.join(SANDBOX_DIR, "test.txt")
if os.path.exists(target_file):
    print("✅ SUCCESS: The file was created inside the sandbox!")
    with open(target_file, "r") as f:
        print(f"File contents: '{f.read()}'")
else:
    print("❌ FAILED: The file was not created.")