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

# 获取API配置
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', '').rstrip('/')

if not api_key or not base_url:
    print("Error: OPENAI_API_KEY or OPENAI_BASE_URL not found in environment variables")
    sys.exit(1)

print(f"\n使用的API配置:")
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:8]}...")

# 尝试不同的API端点
endpoints = [
    '/models',           # 标准OpenAI端点
    '/v1/models',        # 带v1前缀的端点
    '/available_models', # 可能的自定义端点
    '/v1/available_models'
]

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("\n尝试获取可用模型列表...")

for endpoint in endpoints:
    try:
        url = f"{base_url}{endpoint}"
        print(f"\n尝试访问: {url}")
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("成功获取模型列表！")
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                models = data['data']
                print("\n可用的模型:")
                for model in models:
                    if isinstance(model, dict):
                        print(f"- {model.get('id', model)}")
                    else:
                        print(f"- {model}")
            else:
                print("\n响应数据:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            break
        else:
            print("响应内容:", response.text)
            
    except Exception as e:
        print(f"访问 {endpoint} 时出错: {str(e)}")

print("\n提示：如果无法获取模型列表，请：")
print("1. 检查代理服务的管理面板")
print("2. 查看账户余额和权限")
print("3. 联系服务提供商获取支持") 