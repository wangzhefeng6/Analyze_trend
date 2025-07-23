import os
import json
import sys
import dotenv
import argparse
import time
import requests
import urllib3
import winreg  # Windows注册表访问

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from structure import Structure

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 获取项目根目录的.env文件
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
env_path = os.path.join(root_dir, '.env')

if os.path.exists(env_path):
    dotenv.load_dotenv(env_path)
    print(f"Loaded environment from {env_path}", file=sys.stderr)
else:
    print(f"Warning: .env file not found at {env_path}", file=sys.stderr)

# 设置默认的API配置
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'sk-ngUapJAt1q3Dm5d2094aDe236d60422589B9Ab4bAaF933Ab')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# 定义支持的模型
PROXY_MODELS = {
    'deepseek-chat': '适用于代理服务',
    'gpt-3.5-turbo': '需要OpenAI API或代理服务支持',
    'gpt-4': '需要OpenAI API或代理服务支持',
}

template = open(os.path.join(current_dir, "template.txt"), "r", encoding='utf-8').read()
system = open(os.path.join(current_dir, "system.txt"), "r", encoding='utf-8').read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--no-proxy", action="store_true", help="禁用所有代理")
    parser.add_argument("--proxy", type=str, help="指定代理地址 (例如: http://127.0.0.1:7890)")
    parser.add_argument("--test-network", action="store_true", help="测试网络连接")
    return parser.parse_args()

def test_network_connection():
    """测试网络连接和代理状态"""
    print("=" * 50, file=sys.stderr)
    print("网络连接诊断", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # 检查Windows系统代理
    windows_proxy = check_windows_proxy()
    
    # 测试基本网络连接
    test_urls = [
        "https://httpbin.org/ip",  # 获取当前IP
        "https://api.openai.com/v1/models",  # OpenAI API
        "https://www.google.com",  # 基本连接测试
    ]
    
    connection_results = {}
    
    for url in test_urls:
        try:
            print(f"测试连接: {url}", file=sys.stderr)
            response = requests.get(url, timeout=10, verify=False)
            print(f"  ✅ 状态码: {response.status_code}", file=sys.stderr)
            connection_results[url] = "success"
            
            if "httpbin.org/ip" in url:
                try:
                    ip_info = response.json()
                    print(f"  📍 当前IP: {ip_info.get('origin', 'Unknown')}", file=sys.stderr)
                except:
                    pass
        except Exception as e:
            error_msg = str(e)
            connection_results[url] = "failed"
            print(f"  ❌ 失败: {error_msg}", file=sys.stderr)
            
            # 分析错误类型
            if "10054" in error_msg:
                print("    💡 这是Windows连接重置错误，通常由代理不稳定引起", file=sys.stderr)
            elif "ProxyError" in error_msg:
                print("    💡 这是代理连接错误", file=sys.stderr)
            elif "timeout" in error_msg.lower():
                print("    💡 这是网络超时，可能是代理延迟过高", file=sys.stderr)
    
    # 检查环境变量中的代理设置
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
    print("\n代理环境变量:", file=sys.stderr)
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}", file=sys.stderr)
        else:
            print(f"  {var}: 未设置", file=sys.stderr)
    
    # 分析连接模式
    success_count = sum(1 for result in connection_results.values() if result == "success")
    total_count = len(connection_results)
    
    print(f"\n📊 连接统计: {success_count}/{total_count} 成功", file=sys.stderr)
    
    if success_count == 0:
        print("🚨 所有连接都失败，建议:", file=sys.stderr)
        print("   1. 检查科学上网工具是否正常运行", file=sys.stderr)
        print("   2. 尝试切换代理节点", file=sys.stderr)
        print("   3. 使用 --no-proxy 完全禁用代理", file=sys.stderr)
    elif success_count < total_count:
        print("⚠️  部分连接失败，网络不稳定，建议:", file=sys.stderr)
        print("   1. 检查代理服务器负载", file=sys.stderr)
        print("   2. 尝试使用更稳定的代理节点", file=sys.stderr)
        print("   3. 增加重试次数和等待时间", file=sys.stderr)
    
    print("=" * 50, file=sys.stderr)

def configure_proxy_settings(args):
    """配置代理设置"""
    if args.no_proxy:
        print("🚫 禁用所有代理", file=sys.stderr)
        # 清除所有代理环境变量
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        
        # 设置requests的默认行为
        os.environ['NO_PROXY'] = '*'
        
    elif args.proxy:
        print(f"🔄 使用指定代理: {args.proxy}", file=sys.stderr)
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
    else:
        print("🔍 使用当前代理设置", file=sys.stderr)

def create_llm_with_proxy_handling(model_name):
    """创建LLM实例，处理代理相关问题"""
    try:
        # 方式1：尝试正常连接
        print(f"🔗 尝试正常连接模型: {model_name}", file=sys.stderr)
        
        # 首先尝试支持function calling的模型
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                request_timeout=30,  # 增加超时时间
                max_retries=2,       # 减少重试次数
            ).with_structured_output(Structure, method="function_calling")
            
            # 实际测试模型可用性
            print(f"🧪 测试模型 {model_name} function calling 可用性...", file=sys.stderr)
            test_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个AI助手，请简洁回复。"),
                ("human", "请用中文回复'测试成功'")
            ])
            test_chain = test_prompt | llm
            test_response = test_chain.invoke({})
            print(f"✅ 模型 {model_name} (function calling) 测试成功", file=sys.stderr)
            return llm, "function_calling"
            
        except Exception as fc_error:
            # 如果function calling失败，尝试JSON模式
            if "tool use" in str(fc_error) or "function" in str(fc_error).lower():
                print(f"💡 模型 {model_name} 不支持function calling，尝试JSON模式", file=sys.stderr)
                
                # 创建不带function calling的LLM
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.7,
                    request_timeout=30,
                    max_retries=2,
                )
                
                # 测试基本可用性
                print(f"🧪 测试模型 {model_name} JSON模式可用性...", file=sys.stderr)
                test_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个AI助手，请简洁回复。"),
                    ("human", "请用中文回复'测试成功'")
                ])
                test_chain = test_prompt | llm
                test_response = test_chain.invoke({})
                print(f"✅ 模型 {model_name} (JSON模式) 测试成功", file=sys.stderr)
                return llm, "json_mode"
            else:
                raise fc_error
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ 模型 {model_name} 连接失败: {e}", file=sys.stderr)
        
        # 检查是否是one-api的模型不可用错误
        if "无可用渠道" in str(e) or "one_api_error" in str(e):
            print(f"💡 模型 {model_name} 在one-api中未配置，将尝试其他模型", file=sys.stderr)
            raise e  # 直接抛出，不尝试代理处理
        
        # 检查是否是代理相关错误
        proxy_related_errors = [
            'proxy', 'timeout', 'connection', 'ssl', 'certificate', 
            '503', 'service unavailable', 'network', 'tunnel'
        ]
        
        if any(keyword in error_msg for keyword in proxy_related_errors):
            print("🔍 检测到可能的代理相关问题，尝试代理解决方案...", file=sys.stderr)
            
            # 方式2：临时禁用代理重试
            print("🚫 临时禁用代理重试", file=sys.stderr)
            original_proxies = {}
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            
            try:
                # 保存原始代理设置
                for var in proxy_vars:
                    if var in os.environ:
                        original_proxies[var] = os.environ[var]
                        del os.environ[var]
                
                # 设置NO_PROXY
                os.environ['NO_PROXY'] = '*'
                
                # 尝试function calling模式
                try:
                    llm = ChatOpenAI(
                        model=model_name,
                        temperature=0.7,
                        request_timeout=60,
                    ).with_structured_output(Structure, method="function_calling")
                    
                    # 再次测试模型可用性
                    print(f"🧪 禁用代理后测试模型 {model_name} function calling 可用性...", file=sys.stderr)
                    test_prompt = ChatPromptTemplate.from_messages([
                        ("system", "你是一个AI助手，请简洁回复。"),
                        ("human", "请用中文回复'测试成功'")
                    ])
                    test_chain = test_prompt | llm
                    test_response = test_chain.invoke({})
                    
                    print(f"✅ 禁用代理后模型 {model_name} (function calling) 连接成功", file=sys.stderr)
                    return llm, "function_calling"
                    
                except Exception as fc_error2:
                    if "tool use" in str(fc_error2) or "function" in str(fc_error2).lower():
                        print(f"💡 禁用代理后模型 {model_name} 仍不支持function calling，使用JSON模式", file=sys.stderr)
                        
                        llm = ChatOpenAI(
                            model=model_name,
                            temperature=0.7,
                            request_timeout=60,
                        )
                        
                        print(f"✅ 禁用代理后模型 {model_name} (JSON模式) 连接成功", file=sys.stderr)
                        return llm, "json_mode"
                    else:
                        raise fc_error2
                
            except Exception as e2:
                print(f"❌ 禁用代理后仍然失败: {e2}", file=sys.stderr)
                
                # 恢复原始代理设置
                for var, value in original_proxies.items():
                    os.environ[var] = value
                if 'NO_PROXY' in os.environ:
                    del os.environ['NO_PROXY']
                
                raise e  # 抛出原始错误
        else:
            raise e

def check_windows_proxy():
    """检查Windows系统代理设置"""
    try:
        # 访问注册表获取代理设置
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        
        proxy_enable = winreg.QueryValueEx(reg_key, "ProxyEnable")[0]
        if proxy_enable:
            proxy_server = winreg.QueryValueEx(reg_key, "ProxyServer")[0]
            print(f"🔍 检测到Windows系统代理: {proxy_server}", file=sys.stderr)
            return proxy_server
        else:
            print("🔍 Windows系统代理: 未启用", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"🔍 无法检查Windows代理设置: {e}", file=sys.stderr)
        return None
    finally:
        try:
            winreg.CloseKey(reg_key)
        except:
            pass

def disable_system_proxy():
    """临时禁用系统代理"""
    try:
        # 保存原始设置
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                                0, winreg.KEY_ALL_ACCESS)
        
        original_enable = winreg.QueryValueEx(reg_key, "ProxyEnable")[0]
        
        # 禁用代理
        winreg.SetValueEx(reg_key, "ProxyEnable", 0, winreg.REG_DWORD, 0)
        winreg.CloseKey(reg_key)
        
        print("🚫 已临时禁用Windows系统代理", file=sys.stderr)
        return original_enable
        
    except Exception as e:
        print(f"❌ 无法禁用系统代理: {e}", file=sys.stderr)
        return None

def restore_system_proxy(original_enable):
    """恢复原始代理设置"""
    try:
        if original_enable is not None:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                    r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                                    0, winreg.KEY_ALL_ACCESS)
            
            winreg.SetValueEx(reg_key, "ProxyEnable", 0, winreg.REG_DWORD, original_enable)
            winreg.CloseKey(reg_key)
            
            print("🔄 已恢复Windows系统代理设置", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ 无法恢复系统代理: {e}", file=sys.stderr)

def main():
    args = parse_args()
    
    # 如果用户请求网络测试
    if args.test_network:
        test_network_connection()
        return
    
    # 配置代理设置
    configure_proxy_settings(args)
    
    # 优先使用代理服务支持的模型
    model_name = os.environ.get("MODEL_NAME", 'deepseek/deepseek-r1:free')  # 默认使用免费模型
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 可用的模型列表（按优先级排序，免费和代理服务友好的模型在前）
    available_models = [
        'deepseek/deepseek-r1:free', 
        'qwq-32b:free',
        'deepseek-chat', 
        'gpt-3.5-turbo', 
        'gpt-4', 
        'gemini-2.0-flash-001'
    ]
    
    print(f"🎯 目标模型: {model_name}", file=sys.stderr)
    print(f"🔄 备用模型: {available_models}", file=sys.stderr)

    # 修复路径处理
    if args.data.startswith('..'):
        # 如果是相对于当前目录的上级目录的路径
        args.data = os.path.abspath(os.path.join(current_dir, args.data))
    elif not os.path.isabs(args.data):
        # 如果是相对于项目根目录的路径
        args.data = os.path.join(root_dir, args.data)

    if not os.path.exists(args.data):
        raise ValueError(f"Data file not found: {args.data}")

    print(f"📖 读取数据: {args.data}", file=sys.stderr)
    
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data

    print(f'📊 数据加载完成: {len(data)} 篇论文', file=sys.stderr)

    # 尝试连接模型，使用代理处理功能
    llm = None
    model_mode = None
    for model in [model_name] + [m for m in available_models if m != model_name]:
        try:
            llm, mode = create_llm_with_proxy_handling(model)
            model_name = model  # 更新实际使用的模型名
            model_mode = mode
            print(f"🚀 将使用模型: {model_name} ({mode})", file=sys.stderr)
            break
        except Exception as e:
            error_msg = str(e)
            if "无可用渠道" in error_msg or "one_api_error" in error_msg:
                print(f'🔄 模型 {model} 在one-api中未配置，尝试下一个模型', file=sys.stderr)
            else:
                print(f'❌ 模型 {model} 最终连接失败: {e}', file=sys.stderr)
            continue
    
    if llm is None:
        print("❌ 所有模型都无法连接！", file=sys.stderr)
        print("💡 问题诊断:", file=sys.stderr)
        print("   1. one-api服务中可能没有配置任何可用的模型渠道", file=sys.stderr)
        print("   2. 请检查one-api管理面板中的渠道配置", file=sys.stderr)
        print("   3. 确保至少配置了免费模型或常用模型", file=sys.stderr)
        print("   4. 检查API密钥是否正确配置", file=sys.stderr)
        raise RuntimeError("所有模型都无法连接，请检查one-api配置")
    
    # 根据模型模式创建不同的处理链
    if model_mode == "function_calling":
        # 使用结构化输出
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(template=template)
        ])
        chain = prompt_template | llm
        
    else:  # JSON模式
        # 创建JSON格式的prompt
        json_system = system + """

请按照以下JSON格式回复，确保输出是有效的JSON：
{{
    "tldr": "简洁的一句话总结",
    "motivation": "研究动机和背景",
    "method": "主要方法和技术",
    "result": "关键结果和发现",
    "conclusion": "结论和意义"
}}

只输出JSON，不要包含任何其他文字。"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(json_system),
            HumanMessagePromptTemplate.from_template(template=template)
        ])
        chain = prompt_template | llm

    output_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    print(f"📝 输出文件: {output_file}", file=sys.stderr)
    
    for idx, d in enumerate(data):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if model_mode == "function_calling":
                    # 结构化输出模式
                    response: Structure = chain.invoke({
                        "language": language,
                        "content": d['summary']
                    })
                    d['AI'] = response.model_dump()
                else:
                    # JSON模式 - 需要解析JSON
                    response = chain.invoke({
                        "language": language,
                        "content": d['summary']
                    })
                    
                    # 解析JSON响应
                    try:
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        # 提取JSON部分
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            json_str = response_text[json_start:json_end]
                            parsed_response = json.loads(json_str)
                            
                            # 确保所有必需字段存在
                            required_fields = ["tldr", "motivation", "method", "result", "conclusion"]
                            for field in required_fields:
                                if field not in parsed_response:
                                    parsed_response[field] = "信息不完整"
                                    
                            d['AI'] = parsed_response
                        else:
                            raise ValueError("No valid JSON found in response")
                            
                    except (json.JSONDecodeError, ValueError) as json_error:
                        print(f"⚠️ JSON解析失败: {json_error}", file=sys.stderr)
                        d['AI'] = {
                            "tldr": f"AI回复解析失败: {str(response)[:100]}...",
                            "motivation": "解析错误",
                            "method": "解析错误",
                            "result": "解析错误", 
                            "conclusion": "请检查模型输出格式"
                        }
                
                break  # 成功则跳出重试循环
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(f"📄 {d['id']} 第{retry_count}次尝试失败: {error_msg}", file=sys.stderr)
                
                # 检查是否是one-api模型配置错误
                if "无可用渠道" in error_msg:
                    print(f"💡 检测到模型配置问题，跳过重试", file=sys.stderr)
                    retry_count = max_retries  # 直接跳到最大重试次数
                
                # 检查是否是代理或网络相关错误
                network_errors = ["503", "timeout", "connection", "proxy", "ssl", "certificate", "network"]
                if any(keyword in error_msg.lower() for keyword in network_errors):
                    if retry_count < max_retries:
                        wait_time = retry_count * 3  # 增加等待时间
                        print(f"🔄 网络问题，等待 {wait_time} 秒后重试...", file=sys.stderr)
                        time.sleep(wait_time)
                        continue
                
                # 达到最大重试次数
                if retry_count >= max_retries:
                    print(f"❌ {d['id']} 重试 {max_retries} 次后仍然失败", file=sys.stderr)
                    d['AI'] = {
                        "tldr": f"处理错误: {str(e)}",
                        "motivation": "模型处理错误",
                        "method": "模型处理错误", 
                        "result": "模型处理错误",
                        "conclusion": "建议检查模型配置或网络设置"
                    }
        
        # 使用追加模式写入文件，这样即使中途出错也能保存已处理的结果
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"✅ 完成 {idx+1}/{len(data)} (模型: {model_name})", file=sys.stderr)

if __name__ == "__main__":
    main()