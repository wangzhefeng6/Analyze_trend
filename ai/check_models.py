import os
import sys
import json
import requests
import dotenv
from pathlib import Path

# 获取项目根目录的.env文件
current_dir = Path(__file__).parent
root_dir = current_dir.parent
env_path = root_dir / '.env'

if env_path.exists():
    dotenv.load_dotenv(env_path)
    print(f"Loaded environment from {env_path}", file=sys.stderr)
else:
    print(f"Warning: .env file not found at {env_path}", file=sys.stderr)

# 获取并显示所有相关的环境变量
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', '').rstrip('/')
model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

print("\n当前配置:")
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:8]}..." if api_key else "API Key: Not set")
print(f"Default Model: {model_name}")

def test_chat_completion(model):
    """测试特定模型的聊天完成功能"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    print(f"\n测试模型 {model}:")
    print(f"请求URL: {base_url}/chat/completions")
    print(f"请求头: {json.dumps(headers, indent=2, default=str)}")
    print(f"请求体: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text[:500]}...")  # 只显示前500个字符
        
        if response.status_code == 200:
            return True, "可用"
        else:
            error_msg = response.json().get('error', {}).get('message', '未知错误')
            return False, error_msg
            
    except requests.exceptions.RequestException as e:
        return False, f"请求错误: {str(e)}"
    except Exception as e:
        return False, f"其他错误: {str(e)}"

def main():
    # 测试默认模型
    print(f"\n=== 测试默认模型 {model_name} ===")
    success, message = test_chat_completion(model_name)
    if success:
        print(f"✓ {model_name} 测试成功")
    else:
        print(f"✗ {model_name} 测试失败: {message}")
    
    # 测试其他常见模型
    other_models = ['gpt-3.5-turbo', 'gpt-4', 'deepseek-chat', 'claude-2']
    other_models = [m for m in other_models if m != model_name]
    
    if other_models:
        print("\n=== 测试其他模型 ===")
        for model in other_models:
            success, message = test_chat_completion(model)
            if success:
                print(f"✓ {model} 测试成功")
            else:
                print(f"✗ {model} 测试失败: {message}")

if __name__ == "__main__":
    main() 