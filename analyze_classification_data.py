#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 data_analysis_agent 分析 classification_cache.json 文件
分析学术论文分类数据，生成可视化报告
"""

import os
import sys
import json
from pathlib import Path

# 添加 data_analysis_agent 到 Python 路径
sys.path.append(str(Path(__file__).parent / "data_analysis_agent"))

from data_analysis_agent import DataAnalysisAgent
from config.llm_config import LLMConfig

def main():
    """主函数：运行数据分析"""
    
    # 检查数据文件是否存在
    cache_file = "analyze_agent/classification_cache.json"
    if not os.path.exists(cache_file):
        print(f"❌ 错误：找不到文件 {cache_file}")
        print("请先运行 domain_hierarchical_aggregator.py 生成分类数据")
        return
    
    print("🔍 开始分析学术论文分类数据...")
    print(f"📁 数据文件: {cache_file}")
    
    try:
        # 初始化 LLM 配置
        print("⚙️  初始化 LLM 配置...")
        llm_config = LLMConfig()
        
        # 创建数据分析代理
        print("🤖 创建数据分析代理...")
        agent = DataAnalysisAgent(llm_config)
        
        # 定义分析任务
        analysis_prompt = """
请对学术论文分类数据进行全面分析，重点关注以下几个方面：

1. **论文分布分析**：
   - 各领域的论文数量和占比
   - 热门研究领域排名
   - 领域间的关联性分析

2. **关键词分析**：
   - 高频关键词统计
   - 关键词共现网络
   - 新兴研究主题识别

3. **研究方法分析**：
   - 常用研究方法统计
   - 方法应用趋势分析
   - 跨领域方法应用情况

4. **问题分析**：
   - 主要研究问题分类
   - 问题解决难度评估
   - 问题热点变化趋势

5. **可视化要求**：
   - 生成饼图显示领域分布
   - 生成柱状图显示关键词频率
   - 生成热力图显示领域关联
   - 生成时间序列图显示趋势变化

6. **深度洞察**：
   - 识别研究热点和冷门领域
   - 分析跨领域研究趋势
   - 提供研究建议和方向

请生成专业的分析报告，包含数据洞察、可视化图表和 actionable insights。
"""
        
        # 运行分析
        print("📊 开始数据分析...")
        report = agent.analyze(
            user_input=analysis_prompt,
            files=[cache_file]
        )
        
        # 保存分析结果
        output_dir = "data_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 Markdown 报告
        md_file = os.path.join(output_dir, "classification_analysis_report.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Markdown 报告已保存: {md_file}")
        
        # 保存 Word 报告（如果有的话）
        word_file = os.path.join(output_dir, "classification_analysis_report.docx")
        if hasattr(agent, 'save_word_report'):
            agent.save_word_report(report, word_file)
            print(f"📄 Word 报告已保存: {word_file}")
        
        print("✅ 数据分析完成！")
        print(f"📁 结果保存在: {output_dir}/")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 