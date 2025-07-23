import os
import json
import sys
import logging
import argparse
import time
from structure import Structure
from chatglm_model import ChatGLM3
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
template = open(os.path.join(current_dir, "template.txt"), "r", encoding='utf-8').read()
system = open(os.path.join(current_dir, "system.txt"), "r", encoding='utf-8').read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

def process_paper(llm, paper_data, system_prompt, max_retries=3):
    """处理单篇论文"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 构建完整的提示
            full_prompt = f"{system_prompt}\n\n{template.format(content=paper_data['summary'])}"
            
            # 使用结构化输出
            response = llm.structured_output(full_prompt, Structure)
            paper_data['AI'] = response.model_dump()
            logger.info(f"论文 {paper_data['id']} 处理成功")
            return True
            
        except Exception as e:
            retry_count += 1
            logger.error(f"处理论文 {paper_data['id']} 时发生错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(2)  # 等待2秒后重试
            else:
                logger.error(f"论文 {paper_data['id']} 处理失败,已达到最大重试次数")
                paper_data['AI'] = {
                    "tldr": "Error",
                    "motivation": "Error",
                    "method": "Error",
                    "result": "Error",
                    "conclusion": "Error"
                }
                return False

def main():
    args = parse_args()
    language = os.environ.get("LANGUAGE", 'zh-cn')

    logger.info('打开数据文件: %s', args.data)
    
    data = []
    with open(args.data, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data

    # 使用ChatGLM3模型
    logger.info('初始化ChatGLM3模型...')
    try:
        llm = ChatGLM3()
        logger.info('ChatGLM3模型初始化完成')
    except Exception as e:
        logger.error(f'模型初始化失败: {str(e)}')
        sys.exit(1)
    
    # 构建完整的提示模板
    system_prompt = system.format(language=language)
    
    output_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(output_file):
        os.remove(output_file)

    success_count = 0
    for idx, d in enumerate(data):
        logger.info('处理论文 %d/%d - ID: %s', idx+1, len(data), d['id'])
        
        if process_paper(llm, d, system_prompt):
            success_count += 1
        
        # 每处理完一篇论文就保存
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
        logger.info('完成 %d/%d - ID: %s', idx+1, len(data), d['id'])
    
    logger.info(f'处理完成! 成功率: {success_count}/{len(data)} ({success_count/len(data)*100:.2f}%)')

if __name__ == "__main__":
    main()