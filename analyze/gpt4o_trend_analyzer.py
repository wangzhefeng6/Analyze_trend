#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文趋势分析系统
用于处理data目录下的jsonl文件，使用OpenAI API大模型分析论文摘要并统计趋势
"""

import json
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPT4oTrendAnalyzer:
    """使用OpenAI API进行论文趋势分析的类"""
    
    def __init__(self, api_key=None, proxy_url=None, data_dir="data"):
        """
        初始化OpenAI分析器
        
        Args:
            api_key: OpenAI API密钥
            proxy_url: 代理URL（废弃，保留用于兼容性）
            data_dir: 数据目录路径
        """
        # 加载环境变量
        env_file = '.env'
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"已加载环境变量文件: {env_file}")
        
        # API配置
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("未找到API密钥，请设置OPENAI_API_KEY环境变量")
        
        # OpenAI客户端配置
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.dou.chat/v1')
        self.model_name = "openai/gpt-4.1"
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"使用API基础URL: {self.base_url}")
        logger.info(f"使用模型: {self.model_name}")
        
        # 缓存文件路径
        self.cache_file = Path("cache/openai_analysis_cache.pkl")
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # 数据目录
        self.data_dir = data_dir
        
        # 加载缓存
        self.analysis_cache = self._load_cache()
        
        logger.info(f"✅ OpenAI {self.model_name} API客户端初始化成功")
        
        # 分析提示词模板
        self.analysis_prompt = """
你是一名资深的人工智能与计算机科学领域的学术分析专家，拥有深度的技术理解能力。
请对以下学术论文摘要进行全面而精确的技术分析。

**论文摘要**：
{abstract}

**分析要求**：
请严格按照以下维度进行深度分析，确保输出内容的准确性和专业性：

1. **核心问题识别** (problems)：
   - 论文试图解决的主要技术问题或挑战
   - 现有方法的局限性或不足
   - 研究动机和问题背景
   
2. **技术方法** (methods)：
   - 使用的核心算法、模型或技术框架
   - 创新的技术手段或改进方法
   - 实验设计和评估方法
   
3. **应用领域** (domains)：
   - 主要应用场景和目标领域
   - 潜在的应用拓展方向
   - 实际应用价值
   
4. **技术关键词** (keywords)：
   - 提取5-8个最重要的技术术语
   - 包括算法名称、技术概念、评估指标等
   - 优先选择具有代表性的专业术语
   
5. **创新评分** (score)：
   - 评分标准：1分(增量改进) 2分(一般改进) 3分(显著改进) 4分(重要突破) 5分(开创性贡献)
   - 基于技术新颖性、方法创新性、性能提升、实用价值等维度综合评估
   
6. **创新点描述** (innovation)：
   - 总结主要技术贡献和创新亮点
   - 与现有工作的关键区别
   - 对领域发展的潜在影响

**输出格式**：
请严格按照以下JSON格式返回结果，确保字段名称和结构完全匹配：

{{
  "problems": ["具体问题描述1", "具体问题描述2"],
  "methods": ["技术方法1", "技术方法2", "技术方法3"],
  "domains": ["应用领域1", "应用领域2"],
  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
  "score": 数值,
  "innovation": "详细的创新点描述，包括主要贡献和技术亮点"
}}

**注意事项**：
- 请确保分析内容基于摘要的实际内容，避免过度推测
- 技术术语请使用准确的专业表述
- 创新评分需要客观公正，有合理依据
- 如果摘要信息不足，请在相应字段中标注"信息不足"
"""
        
        # arXiv分类映射
        self.category_mapping = {
            'cs.CV': '计算机视觉',
            'cs.AI': '人工智能',
            'cs.LG': '机器学习',
            'cs.CL': '计算语言学',
            'cs.RO': '机器人学',
            'cs.GR': '计算机图形学',
            'cs.IR': '信息检索',
            'cs.HC': '人机交互',
            'cs.NE': '神经与进化计算',
            'cs.MA': '多智能体系统',
            'cs.SE': '软件工程',
            'eess.IV': '图像与视频处理',
            'eess.SP': '信号处理',
            'stat.ML': '统计机器学习',
            'math.OC': '优化与控制'
        }
    
    def _load_cache(self) -> Dict:
        """
        加载分析缓存
        
        Returns:
            缓存字典
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"已加载缓存文件: {self.cache_file}")
                return cache
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
        
        return {}
    
    def _save_cache(self):
        """保存分析缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.analysis_cache, f)
            logger.debug("缓存已保存")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
        
    def load_jsonl_files(self, start_date: str = None, end_date: str = None, categories: List[str] = None) -> List[Dict]:
        """
        加载指定时间范围和类别的jsonl文件
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            categories: 要筛选的arXiv类别列表 (如 ['cs.CV', 'cs.AI'])
            
        Returns:
            论文数据列表
        """
        papers = []
        
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            # 尝试相对于当前脚本的路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            alt_data_dir = os.path.join(parent_dir, 'data')
            
            if os.path.exists(alt_data_dir):
                self.data_dir = alt_data_dir
                logger.info(f"使用替代数据目录: {self.data_dir}")
            else:
                # 提供详细的错误信息
                current_dir = os.getcwd()
                possible_paths = [
                    os.path.abspath(self.data_dir),
                    alt_data_dir,
                    os.path.join(current_dir, 'data'),
                    os.path.join(current_dir, '..', 'data')
                ]
                
                error_msg = f"""
数据目录未找到！请检查以下可能的路径：
当前工作目录: {current_dir}
尝试的路径:
{chr(10).join(f'  - {path} {"✓" if os.path.exists(path) else "✗"}' for path in possible_paths)}

请确保data目录存在并包含JSONL文件，或者：
1. 在当前目录创建data目录
2. 设置正确的data_dir参数
3. 确保工作目录正确
"""
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # 获取所有jsonl文件
        try:
            jsonl_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jsonl')]
            jsonl_files.sort()
        except Exception as e:
            raise FileNotFoundError(f"无法访问数据目录 {self.data_dir}: {e}")
        
        for filename in jsonl_files:
            # 提取日期
            date_str = filename.replace('.jsonl', '')
            try:
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
                
            # 检查日期范围
            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if file_date < start_dt:
                    continue
                    
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if file_date > end_dt:
                    continue
            
            # 加载文件
            filepath = os.path.join(self.data_dir, filename)
            try:
                file_papers = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            paper = json.loads(line.strip())
                            paper['date'] = date_str
                            
                            # 类别筛选
                            if categories:
                                paper_categories = paper.get('categories', [])
                                # 检查论文是否属于指定类别
                                if not any(cat in paper_categories for cat in categories):
                                    continue
                            
                            file_papers.append(paper)
                
                papers.extend(file_papers)
                logger.info(f"已加载文件: {filename} ({len(file_papers)} 篇论文)")
            except Exception as e:
                logger.error(f"加载文件 {filename} 失败: {e}")
                
        # 统计类别信息
        if categories:
            logger.info(f"按类别筛选后加载了 {len(papers)} 篇论文 (筛选类别: {categories})")
        else:
            logger.info(f"总共加载了 {len(papers)} 篇论文")
        
        return papers
    
    def get_available_categories(self, start_date: str = None, end_date: str = None) -> Dict[str, int]:
        """
        获取数据中所有可用的arXiv类别及其论文数量
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            类别字典 {类别: 论文数量}
        """
        papers = self.load_jsonl_files(start_date, end_date, categories=None)
        
        category_counts = defaultdict(int)
        for paper in papers:
            categories = paper.get('categories', [])
            for category in categories:
                category_counts[category] += 1
        
        # 按论文数量降序排序
        sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"找到 {len(sorted_categories)} 个不同的arXiv类别")
        return sorted_categories
    
    def filter_papers_by_category(self, papers: List[Dict], categories: List[str]) -> List[Dict]:
        """
        按类别筛选论文
        
        Args:
            papers: 论文列表
            categories: 要筛选的类别列表
            
        Returns:
            筛选后的论文列表
        """
        if not categories:
            return papers
        
        filtered_papers = []
        for paper in papers:
            paper_categories = paper.get('categories', [])
            if any(cat in paper_categories for cat in categories):
                filtered_papers.append(paper)
        
        logger.info(f"按类别筛选: {len(papers)} -> {len(filtered_papers)} 篇论文")
        return filtered_papers
    
    def analyze_paper_with_gpt4o(self, abstract: str) -> Dict:
        """
        使用OpenAI API大模型分析单篇论文摘要 - 优化版本
        
        Args:
            abstract: 论文摘要
            
        Returns:
            分析结果字典
        """
        if not abstract or len(abstract.strip()) < 50:
            return self._get_default_analysis()
            
        try:
            prompt = self.analysis_prompt.format(abstract=abstract[:2000])  # 限制摘要长度以减少处理时间
            
            # 使用OpenAI客户端调用API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名专业的学术分析专家，请严格按照JSON格式返回分析结果。你的回复必须是一个有效的JSON对象，不要包含任何其他文本或解释。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 降低随机性以获得更一致的结果
                max_tokens=800    # 减少token数以加快响应
            )
            
            result_data = completion.model_dump()
            
            # 检查响应格式
            if 'choices' not in result_data or not result_data['choices']:
                logger.error("API响应格式异常")
                return self._get_default_analysis()
                
            result_text = result_data['choices'][0]['message']['content'].strip()
            
            # 尝试解析JSON - 改进版本
            try:
                # 提取JSON部分（去除可能的markdown标记）
                if '```json' in result_text:
                    json_part = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    json_part = result_text.split('```')[1]
                elif '{' in result_text and '}' in result_text:
                    # 提取第一个完整的JSON对象
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    json_part = result_text[start_idx:end_idx]
                else:
                    json_part = result_text
                    
                result = json.loads(json_part.strip())
                
                # 验证和清理增强版分析结果
                cleaned_result = {}
                
                # 直接使用简化的字段
                cleaned_result['problems'] = result.get('problems', [])[:5]
                cleaned_result['methods'] = result.get('methods', [])[:5]
                cleaned_result['domains'] = result.get('domains', [])[:3]
                cleaned_result['keywords'] = result.get('keywords', [])[:10]
                cleaned_result['innovation'] = result.get('innovation', '')
                
                # 验证分数
                score = result.get('score', 0)
                if isinstance(score, (int, float)) and 0 <= score <= 5:
                    cleaned_result['score'] = float(score)
                else:
                    cleaned_result['score'] = 0.0
                
                return cleaned_result
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"JSON解析失败，尝试简化解析: {e}")
                # 尝试简化的文本解析
                return self._parse_text_fallback(result_text)
                
        except Exception as e:
            logger.error(f"OpenAI API分析失败: {e}")
            return self._get_default_analysis()
    
    def _parse_text_fallback(self, text: str) -> Dict:
        """
        文本解析备用方案
        
        Args:
            text: API返回的文本
            
        Returns:
            解析结果
        """
        result = self._get_default_analysis()
        
        try:
            # 尝试从文本中提取信息
            text_lower = text.lower()
            
            # 提取关键词（简单的启发式方法）
            keywords = []
            if '关键词' in text or 'keyword' in text_lower:
                # 查找可能的关键词列表
                lines = text.split('\n')
                for line in lines:
                    if '关键词' in line or 'keyword' in line.lower():
                        # 提取冒号后的内容
                        if ':' in line:
                            keyword_text = line.split(':', 1)[1].strip()
                            keywords = [k.strip() for k in keyword_text.split(',') if k.strip()][:5]
                        break
            
            if keywords:
                result['keywords'] = keywords
            
            # 尝试提取分数
            import re
            score_pattern = r'[分评]\s*[：:]\s*([0-9.]+)'
            score_match = re.search(score_pattern, text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    if 0 <= score <= 5:
                        result['score'] = score
                except ValueError:
                    pass
                    
        except Exception as e:
            logger.debug(f"文本解析备用方案失败: {e}")
        
        return result
    
    def _get_default_analysis(self) -> Dict:
        """返回默认分析结果"""
        return {
            'problems': [],
            'methods': [],
            'domains': [],
            'score': 0.0,
            'keywords': [],
            'innovation': ''
        }
    
    def batch_analyze_papers(self, papers: List[Dict], max_workers: int = 10, 
                           batch_size: int = 50) -> List[Dict]:
        """
        批量分析论文 - 优化版本
        
        Args:
            papers: 论文列表
            max_workers: 最大并发数 (提高到10)
            batch_size: 批次大小 (减少到50以提高并发效率)
            
        Returns:
            分析结果列表
        """
        results = []
        
        # 使用类的缓存
        cache = self.analysis_cache.copy()
        
        # 预处理：分离需要分析的论文和已缓存的论文
        papers_to_analyze = []
        cached_results = []
        
        for paper in papers:
            paper_id = paper.get('id', '')
            abstract = paper.get('summary', '').strip()
            
            if not abstract:
                # 添加默认结果给没有摘要的论文
                result = paper.copy()
                result.update(self._get_default_analysis())
                cached_results.append(result)
                continue
                
            # 检查缓存
            if paper_id in cache:
                result = cache[paper_id].copy()
                result.update(paper)
                cached_results.append(result)
            else:
                papers_to_analyze.append(paper)
        
        total_papers = len(papers)
        cached_count = len(cached_results)
        analyze_count = len(papers_to_analyze)
        
        logger.info(f"总论文数: {total_papers}, 缓存命中: {cached_count}, 需要分析: {analyze_count}")
        
        # 如果所有论文都已缓存
        if not papers_to_analyze:
            logger.info("所有论文都已缓存，直接返回结果")
            return cached_results
        
        # 多线程批量分析
        analyzed_results = []
        processed = 0
        
        # 使用更大的线程池进行并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 批次提交任务
            batch_futures = []
            
            for i in range(0, len(papers_to_analyze), batch_size):
                batch_papers = papers_to_analyze[i:i + batch_size]
                
                # 为每个批次中的每篇论文提交单独的任务
                for paper in batch_papers:
                    abstract = paper.get('summary', '').strip()
                    future = executor.submit(self._analyze_single_paper_with_retry, paper, abstract)
                    batch_futures.append(future)
            
            # 收集所有结果
            for future in as_completed(batch_futures):
                try:
                    result = future.result()
                    if result:
                        analyzed_results.append(result)
                        
                        # 更新缓存
                        paper_id = result.get('id', '')
                        if paper_id:
                            analysis_data = {
                                'problems': result.get('problems', []),
                                'methods': result.get('methods', []),
                                'domains': result.get('domains', []),
                                'score': result.get('score', 0.0),
                                'keywords': result.get('keywords', [])
                            }
                            cache[paper_id] = analysis_data
                        
                        processed += 1
                        if processed % 5 == 0:  # 更频繁的进度报告
                            progress = (cached_count + processed) / total_papers * 100
                            logger.info(f"已处理 {cached_count + processed}/{total_papers} 篇论文 ({progress:.1f}%)")
                            
                except Exception as exc:
                    logger.error(f"批次处理异常: {exc}")
                    processed += 1
        
        # 合并所有结果
        all_results = cached_results + analyzed_results
        
        # 更新类的缓存并保存
        try:
            self.analysis_cache.update(cache)
            self._save_cache()
            logger.info(f"缓存已更新，包含 {len(self.analysis_cache)} 个分析结果")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
        
        logger.info(f"批量分析完成，共处理 {len(all_results)} 篇论文")
        return all_results
    
    def _analyze_single_paper_with_retry(self, paper: Dict, abstract: str, max_retries: int = 3) -> Dict:
        """
        分析单篇论文，带重试机制
        
        Args:
            paper: 论文数据
            abstract: 论文摘要
            max_retries: 最大重试次数
            
        Returns:
            分析结果
        """
        for attempt in range(max_retries):
            try:
                analysis = self.analyze_paper_with_gpt4o(abstract)
                
                # 合并结果
                result = paper.copy()
                result.update(analysis)
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 0.5  # 递增等待时间
                    logger.warning(f"论文 {paper.get('id', 'unknown')} 分析失败 (尝试 {attempt + 1}/{max_retries}): {e}, {wait_time}秒后重试")
                    time.sleep(wait_time)
                else:
                    logger.error(f"论文 {paper.get('id', 'unknown')} 分析最终失败: {e}")
                    # 返回默认结果
                    result = paper.copy()
                    result.update(self._get_default_analysis())
                    return result
        
        # 如果所有重试都失败，返回默认结果
        result = paper.copy()
        result.update(self._get_default_analysis())
        return result
    
    def calculate_trends(self, analyzed_papers: List[Dict]) -> Dict:
        """
        计算趋势统计
        
        Args:
            analyzed_papers: 分析后的论文列表
            
        Returns:
            趋势统计结果
        """
        trends = {
            'temporal_trends': defaultdict(lambda: defaultdict(list)),
            'category_trends': defaultdict(lambda: defaultdict(list)),
            'keyword_trends': defaultdict(lambda: defaultdict(int)),
            'method_trends': defaultdict(lambda: defaultdict(int)),
            'problem_trends': defaultdict(lambda: defaultdict(int)),
            'innovation_trends': defaultdict(list),
            'category_distribution': defaultdict(int),
            'daily_stats': defaultdict(lambda: {
                'total_papers': 0,
                'avg_innovation_score': 0.0,
                'top_keywords': [],
                'top_methods': [],
                'top_problems': []
            })
        }
        
        # 按日期和类别统计
        for paper in analyzed_papers:
            date = paper.get('date', '')
            categories = paper.get('categories', [])
            keywords = paper.get('keywords', [])
            methods = paper.get('methods', [])
            problems = paper.get('problems', [])
            score = paper.get('score', 0.0)
            
            # 处理类别
            main_category = categories[0] if categories else 'unknown'
            category_name = self.category_mapping.get(main_category, main_category)
            
            # 时间趋势
            trends['temporal_trends'][date][category_name].append(paper)
            trends['innovation_trends'][date].append(score)
            
            # 类别趋势
            trends['category_trends'][category_name][date].append(paper)
            trends['category_distribution'][category_name] += 1
            
            # 关键词趋势
            for keyword in keywords:
                if keyword:
                    trends['keyword_trends'][date][keyword] += 1
            
            # 方法趋势
            for method in methods:
                if method:
                    trends['method_trends'][date][method] += 1
            
            # 问题趋势
            for problem in problems:
                if problem:
                    trends['problem_trends'][date][problem] += 1
            
            # 每日统计
            daily_stat = trends['daily_stats'][date]
            daily_stat['total_papers'] += 1
            
            # 更新关键词、方法、问题统计
            for keyword in keywords:
                if keyword:
                    daily_stat.setdefault('keywords', Counter())[keyword] += 1
            for method in methods:
                if method:
                    daily_stat.setdefault('methods', Counter())[method] += 1
            for problem in problems:
                if problem:
                    daily_stat.setdefault('problems', Counter())[problem] += 1
        
        # 计算平均创新分数和Top项目
        for date, stats in trends['daily_stats'].items():
            scores = trends['innovation_trends'][date]
            if scores:
                stats['avg_innovation_score'] = np.mean(scores)
            
            # Top关键词、方法、问题
            if 'keywords' in stats:
                stats['top_keywords'] = stats['keywords'].most_common(10)
            if 'methods' in stats:
                stats['top_methods'] = stats['methods'].most_common(10)
            if 'problems' in stats:
                stats['top_problems'] = stats['problems'].most_common(10)
        
        return dict(trends)
    
    def analyze_by_category(self, analyzed_papers: List[Dict], target_categories: List[str] = None) -> Dict:
        """
        按类别进行深度分析
        
        Args:
            analyzed_papers: 分析后的论文列表
            target_categories: 目标类别列表，None表示分析所有类别
            
        Returns:
            按类别的分析结果
        """
        category_analysis = defaultdict(lambda: {
            'total_papers': 0,
            'avg_innovation_score': 0.0,
            'temporal_trend': defaultdict(int),
            'top_keywords': Counter(),
            'top_methods': Counter(),
            'top_problems': Counter(),
            'innovation_distribution': [],
            'daily_avg_scores': defaultdict(list),
            'paper_details': []
        })
        
        # 筛选论文
        if target_categories:
            filtered_papers = self.filter_papers_by_category(analyzed_papers, target_categories)
        else:
            filtered_papers = analyzed_papers
        
        # 按类别分组分析
        for paper in filtered_papers:
            categories = paper.get('categories', [])
            
            for category in categories:
                if target_categories and category not in target_categories:
                    continue
                    
                category_name = self.category_mapping.get(category, category)
                analysis = category_analysis[category_name]
                
                # 基础统计
                analysis['total_papers'] += 1
                
                # 时间趋势
                date = paper.get('date', '')
                analysis['temporal_trend'][date] += 1
                
                # 关键词、方法、问题统计
                for keyword in paper.get('keywords', []):
                    if keyword:
                        analysis['top_keywords'][keyword] += 1
                
                for method in paper.get('methods', []):
                    if method:
                        analysis['top_methods'][method] += 1
                
                for problem in paper.get('problems', []):
                    if problem:
                        analysis['top_problems'][problem] += 1
                
                # 创新分数
                score = paper.get('score', 0.0)
                analysis['innovation_distribution'].append(score)
                analysis['daily_avg_scores'][date].append(score)
                
                # 论文详情
                analysis['paper_details'].append({
                    'id': paper.get('id', ''),
                    'title': paper.get('title', ''),
                    'date': date,
                    'score': score,
                    'categories': categories,
                    'keywords': paper.get('keywords', []),
                    'methods': paper.get('methods', []),
                    'problems': paper.get('problems', [])
                })
        
        # 计算统计数据
        for category_name, analysis in category_analysis.items():
            if analysis['innovation_distribution']:
                analysis['avg_innovation_score'] = np.mean(analysis['innovation_distribution'])
                analysis['std_innovation_score'] = np.std(analysis['innovation_distribution'])
                analysis['min_innovation_score'] = min(analysis['innovation_distribution'])
                analysis['max_innovation_score'] = max(analysis['innovation_distribution'])
            
            # 计算每日平均分数
            daily_scores = {}
            for date, scores in analysis['daily_avg_scores'].items():
                if scores:
                    daily_scores[date] = np.mean(scores)
            analysis['daily_avg_scores'] = daily_scores
            
            # 转换Counter为普通字典，便于JSON序列化
            analysis['top_keywords'] = dict(analysis['top_keywords'].most_common(20))
            analysis['top_methods'] = dict(analysis['top_methods'].most_common(15))
            analysis['top_problems'] = dict(analysis['top_problems'].most_common(15))
            analysis['temporal_trend'] = dict(analysis['temporal_trend'])
        
        return dict(category_analysis)
    
    def compare_categories(self, analyzed_papers: List[Dict], categories: List[str]) -> Dict:
        """
        比较不同类别之间的差异
        
        Args:
            analyzed_papers: 分析后的论文列表
            categories: 要比较的类别列表
            
        Returns:
            类别比较结果
        """
        comparison = {
            'categories': categories,
            'comparison_metrics': {},
            'category_details': {}
        }
        
        category_analysis = self.analyze_by_category(analyzed_papers, categories)
        
        # 比较指标
        metrics = ['total_papers', 'avg_innovation_score', 'std_innovation_score']
        
        for metric in metrics:
            comparison['comparison_metrics'][metric] = {}
            for category in categories:
                category_name = self.category_mapping.get(category, category)
                if category_name in category_analysis:
                    value = category_analysis[category_name].get(metric, 0)
                    comparison['comparison_metrics'][metric][category_name] = value
        
        # 详细数据
        comparison['category_details'] = category_analysis
        
        # 计算相关性和差异
        if len(categories) >= 2:
            scores_by_category = {}
            for category in categories:
                category_name = self.category_mapping.get(category, category)
                if category_name in category_analysis:
                    scores_by_category[category_name] = category_analysis[category_name]['innovation_distribution']
            
            # 统计显著性测试可以在这里添加
            comparison['statistical_summary'] = {
                'total_categories_compared': len(scores_by_category),
                'categories_with_data': list(scores_by_category.keys())
            }
        
        return comparison
    
    def save_analysis_results(self, analyzed_papers: List[Dict], 
                            trends: Dict, output_dir: str = "analysis_results"):
        """
        保存分析结果
        
        Args:
            analyzed_papers: 分析后的论文列表
            trends: 趋势统计结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存分析后的论文数据
        papers_file = os.path.join(output_dir, 'analyzed_papers.json')
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(analyzed_papers, f, ensure_ascii=False, indent=2)
        
        # 保存趋势数据
        trends_file = os.path.join(output_dir, 'trends_analysis.json')
        
        # 将defaultdict转换为普通dict以便JSON序列化
        def convert_defaultdict(obj):
            if isinstance(obj, defaultdict):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_defaultdict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_defaultdict(item) for item in obj]
            elif isinstance(obj, Counter):
                return dict(obj)
            else:
                return obj
        
        trends_serializable = convert_defaultdict(trends)
        
        with open(trends_file, 'w', encoding='utf-8') as f:
            json.dump(trends_serializable, f, ensure_ascii=False, indent=2)
        
        # 创建CSV统计报告
        self._create_csv_reports(analyzed_papers, trends, output_dir)
        
        logger.info(f"分析结果已保存到 {output_dir} 目录")
    
    def _create_csv_reports(self, analyzed_papers: List[Dict], 
                          trends: Dict, output_dir: str):
        """创建CSV格式的统计报告"""
        
        # 1. 每日趋势报告
        daily_data = []
        for date, stats in trends['daily_stats'].items():
            daily_data.append({
                'date': date,
                'total_papers': stats['total_papers'],
                'avg_innovation_score': stats['avg_innovation_score'],
                'top_keywords': ', '.join([k for k, _ in stats.get('top_keywords', [])[:5]]),
                'top_methods': ', '.join([k for k, _ in stats.get('top_methods', [])[:5]]),
                'top_problems': ', '.join([k for k, _ in stats.get('top_problems', [])[:5]])
            })
        
        df_daily = pd.DataFrame(daily_data)
        df_daily.to_csv(os.path.join(output_dir, 'daily_trends.csv'), 
                       index=False, encoding='utf-8')
        
        # 2. 类别分布报告
        category_data = []
        for category, count in trends['category_distribution'].items():
            category_data.append({
                'category': category,
                'paper_count': count,
                'percentage': count / len(analyzed_papers) * 100
            })
        
        df_category = pd.DataFrame(category_data)
        df_category = df_category.sort_values('paper_count', ascending=False)
        df_category.to_csv(os.path.join(output_dir, 'category_distribution.csv'), 
                          index=False, encoding='utf-8')
        
        # 3. 关键词趋势报告
        keyword_trends_data = []
        for date, keywords in trends['keyword_trends'].items():
            for keyword, count in keywords.items():
                keyword_trends_data.append({
                    'date': date,
                    'keyword': keyword,
                    'count': count
                })
        
        df_keywords = pd.DataFrame(keyword_trends_data)
        df_keywords.to_csv(os.path.join(output_dir, 'keyword_trends.csv'), 
                          index=False, encoding='utf-8')
        
        logger.info("CSV报告已生成")

    def run_full_analysis(self, start_date: str = None, end_date: str = None,
                         max_workers: int = 15, categories: List[str] = None,
                         include_category_analysis: bool = True) -> Tuple[List[Dict], Dict]:
        """
        运行完整的分析流程
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_workers: 最大并发数
            categories: 要分析的arXiv类别列表
            include_category_analysis: 是否包含详细的类别分析
            
        Returns:
            (分析后的论文列表, 趋势统计结果)
        """
        logger.info("开始论文趋势分析...")
        
        # 1. 加载数据
        logger.info("正在加载论文数据...")
        papers = self.load_jsonl_files(start_date, end_date, categories)
        
        if not papers:
            logger.warning("没有找到符合条件的论文数据")
            return [], {}
        
        # 2. 批量分析
        logger.info("开始使用OpenAI/gpt-4o大模型分析论文...")
        analyzed_papers = self.batch_analyze_papers(papers, max_workers)
        
        # 3. 计算基础趋势
        logger.info("正在计算趋势统计...")
        trends = self.calculate_trends(analyzed_papers)
        
        # 4. 添加类别分析（如果启用）
        if include_category_analysis:
            logger.info("进行详细类别分析...")
            trends['category_analysis'] = self.analyze_by_category(analyzed_papers, categories)
            
            # 如果指定了多个类别，进行比较分析
            if categories and len(categories) > 1:
                logger.info("进行类别比较分析...")
                trends['category_comparison'] = self.compare_categories(analyzed_papers, categories)
        
        # 5. 保存结果
        logger.info("正在保存分析结果...")
        self.save_analysis_results(analyzed_papers, trends)
        
        logger.info("OpenAI/gpt-4o论文趋势分析完成！")
        return analyzed_papers, trends
    
    def run_category_focused_analysis(self, categories: List[str], start_date: str = None, 
                                     end_date: str = None, max_workers: int = 15) -> Dict:
        """
        运行专注于特定类别的分析
        
        Args:
            categories: 要分析的arXiv类别列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_workers: 最大并发数
            
        Returns:
            类别分析结果
        """
        logger.info(f"开始针对类别 {categories} 的专项分析...")
        
        # 获取可用类别信息
        available_categories = self.get_available_categories(start_date, end_date)
        
        # 验证请求的类别是否存在
        valid_categories = [cat for cat in categories if cat in available_categories]
        if not valid_categories:
            logger.warning(f"指定的类别 {categories} 在数据中未找到")
            return {
                'error': 'No valid categories found',
                'available_categories': available_categories
            }
        
        # 运行完整分析
        analyzed_papers, trends = self.run_full_analysis(
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            categories=valid_categories,
            include_category_analysis=True
        )
        
        # 返回类别专项结果
        return {
            'analyzed_papers': analyzed_papers,
            'basic_trends': trends,
            'category_analysis': trends.get('category_analysis', {}),
            'category_comparison': trends.get('category_comparison', {}),
            'available_categories': available_categories,
            'analyzed_categories': valid_categories
        }

    def summarize_trends_with_gpt4o(self, trends: Dict, categories: List[str] = None, start_date: str = None, end_date: str = None) -> str:
        """
        使用GPT-4o对统计结果进行智能总结
        Args:
            trends: 统计结果字典
            categories: 用户选择的类别
            start_date: 起始日期
            end_date: 结束日期
        Returns:
            智能总结文本
        """
        # 构建统计摘要
        total_papers = sum(trends.get('category_distribution', {}).values())
        avg_score = 0.0
        if 'innovation_trends' in trends and trends['innovation_trends']:
            all_scores = []
            for v in trends['innovation_trends'].values():
                all_scores.extend(v)
            if all_scores:
                avg_score = float(np.mean(all_scores))
        top_keywords = []
        if 'keyword_trends' in trends and trends['keyword_trends']:
            keyword_counter = Counter()
            for day in trends['keyword_trends']:
                keyword_counter.update(trends['keyword_trends'][day])
            top_keywords = [k for k, v in keyword_counter.most_common(10)]
        # 构建优化的prompt
        date_range = f"{start_date or '最近'} 到 {end_date or '当前'}" if start_date or end_date else "当前时间段"
        category_info = f"，聚焦类别：{', '.join(categories)}" if categories else ""
        
        prompt = f"""
你是一名顶级的学术研究趋势分析专家，请基于以下统计数据对 {date_range}{category_info} 的学术论文进行深度趋势分析。

**统计数据摘要**：
📊 论文总数：{total_papers}
🏆 平均创新分数：{avg_score:.2f}/5.0
🔥 热门关键词：{', '.join(top_keywords[:10]) if top_keywords else '暂无数据'}
📈 类别分布：{dict(list(trends.get('category_distribution', {}).items())[:8])}

**分析要求**：
请从以下维度进行专业分析，输出结构化的中文报告：

1. **总体趋势概览**
   - 研究活跃度评估
   - 创新水平总体特征
   - 跨领域融合情况

2. **技术热点识别**
   - 核心技术关键词分析
   - 新兴技术方向识别
   - 技术成熟度评估

3. **研究焦点分析**
   - 主要研究问题和挑战
   - 解决方案的技术路径
   - 应用场景的拓展情况

4. **创新亮点总结**
   - 高分论文的共同特征
   - 突破性技术或方法
   - 具有影响力的研究方向

5. **未来发展预测**
   - 基于当前趋势的发展预测
   - 潜在的研究机会
   - 技术发展的可能方向

**输出格式**：
请用专业且易懂的中文撰写分析报告，每个维度用清晰的段落组织，适当使用表情符号增强可读性。
报告应当客观、准确，基于数据得出结论，避免过度推测。
"""
        
        # 尝试不同的模型名称，以适配不同的API端点
        models_to_try = [
            self.model_name,  # 当前配置的模型
            "gpt-4o",         # 标准OpenAI模型名
            "gpt-4",          # 备用模型
            "gpt-3.5-turbo"   # 最后备用
        ]
        
        for model in models_to_try:
            try:
                logger.info(f"尝试使用模型: {model}")
                # 调用大模型
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一名资深的学术研究趋势分析专家，具有深厚的人工智能、计算机科学和跨学科研究背景。你擅长从大量统计数据中提炼有价值的趋势洞察，能够识别技术发展的规律和未来方向，并用清晰专业的语言表达复杂的学术观点。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                logger.info(f"成功使用模型 {model} 生成总结")
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"使用模型 {model} 失败: {e}")
                if model == models_to_try[-1]:  # 最后一个模型也失败了
                    error_msg = f"""
智能总结生成失败。可能的原因：
1. API配置问题 - 请检查 .env 文件中的配置
2. 模型不可用 - 当前API端点可能不支持指定的模型
3. API额度不足 - 请检查API账户余额

当前配置：
- API基础URL: {self.base_url}
- 尝试的模型: {models_to_try}

建议解决方案：
1. 确认 .env 文件包含正确的 OPENAI_API_KEY 和 OPENAI_BASE_URL
2. 如果使用第三方API，请确认支持的模型名称
3. 可以尝试直接使用 'gpt-3.5-turbo' 或 'gpt-4' 等标准模型名

最后一个错误: {e}
"""
                    raise Exception(error_msg)
                continue
        
        # 理论上不会到达这里，但为了安全起见
        return "抱歉，智能总结生成失败，请检查API配置。"






def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenAI/gpt-4o论文趋势分析 - 支持类别筛选')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--api-key', help='OpenAI API Key')
    parser.add_argument('--data-dir', default='data', help='数据目录')
    parser.add_argument('--max-workers', type=int, default=15, help='最大并发数')
    parser.add_argument('--categories', nargs='+', help='要分析的arXiv类别 (如: cs.CV cs.AI cs.LG)')
    parser.add_argument('--list-categories', action='store_true', help='列出所有可用类别')
    parser.add_argument('--category-only', action='store_true', help='只进行类别专项分析')
    parser.add_argument('--summarize-trends', action='store_true', help='用大模型对统计结果进行智能总结')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = GPT4oTrendAnalyzer(
            api_key=args.api_key,
            data_dir=args.data_dir
        )
        
               
        # 如果只是列出类别
        if args.list_categories:
            print("📊 获取可用类别...")
            categories = analyzer.get_available_categories(args.start_date, args.end_date)
            print(f"\n📈 找到 {len(categories)} 个类别:\n")
            
            for i, (category, count) in enumerate(categories.items(), 1):
                category_name = analyzer.category_mapping.get(category, category)
                print(f"{i:3d}. {category:12s} ({category_name:20s}): {count:4d} 篇")
            return
        
        # 类别专项分析
        if args.category_only and args.categories:
            print(f"🎯 进行类别专项分析: {args.categories}")
            result = analyzer.run_category_focused_analysis(
                categories=args.categories,
                start_date=args.start_date,
                end_date=args.end_date,
                max_workers=args.max_workers
            )
            
            if 'error' in result:
                print(f"❌ {result['error']}")
                print("可用类别:", list(result['available_categories'].keys())[:10])
                return
            
            print(f"\n✅ 类别分析完成！")
            print(f"分析论文: {len(result['analyzed_papers'])} 篇")
            print(f"分析类别: {result['analyzed_categories']}")
            
            # 显示各类别详情
            for category_name, analysis in result['category_analysis'].items():
                print(f"\n📊 {category_name}:")
                print(f"  论文数量: {analysis['total_papers']}")
                print(f"  平均创新分数: {analysis['avg_innovation_score']:.2f}")
                print(f"  热门关键词: {list(analysis['top_keywords'].keys())[:5]}")
                print(f"  主要方法: {list(analysis['top_methods'].keys())[:3]}")
        
        else:
            # 常规分析（可选择性筛选类别）
            print("🚀 开始分析...")
            if args.categories:
                print(f"筛选类别: {args.categories}")
            
            analyzed_papers, trends = analyzer.run_full_analysis(
                start_date=args.start_date,
                end_date=args.end_date,
                max_workers=args.max_workers,
                categories=args.categories
            )
            
            print(f"\n✅ 分析完成！")
            print(f"处理论文: {len(analyzed_papers)} 篇")
            print(f"涉及类别: {len(trends['category_distribution'])} 个")
            
            # 显示类别分布
            print(f"\n📈 类别分布 (前10):")
            category_dist = trends['category_distribution']
            for i, (category, count) in enumerate(list(category_dist.items())[:10], 1):
                percentage = count / len(analyzed_papers) * 100
                print(f"{i:2d}. {category:20s}: {count:4d} 篇 ({percentage:5.1f}%)")
            
            # 如果有类别分析结果
            if 'category_analysis' in trends:
                print(f"\n🔬 详细类别分析已完成")
                
            if 'category_comparison' in trends:
                print(f"📊 类别比较分析已完成")
            
            print(f"\n💾 详细结果已保存到 analysis_results 目录")
            
            if args.summarize_trends:
                print("\n🤖 正在用大模型智能总结趋势...")
                summary = analyzer.summarize_trends_with_gpt4o(
                    trends,
                    categories=args.categories,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                print("\n📋 智能总结：\n" + summary)
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()