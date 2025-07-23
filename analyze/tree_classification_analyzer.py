#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文树形分类分析系统
基于两步骤的细粒度分类分析：
1. 第一步：使用大模型对论文进行细分分类，保存分类路径
2. 第二步：根据分类路径聚合到叶子节点，分析统计每个叶子节点
"""

import json
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Set, Optional
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from threading import Lock
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tree_classification.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入动态语义合并器
try:
    from dynamic_semantic_merger import DynamicSemanticMerger
except ImportError:
    DynamicSemanticMerger = None
    print("⚠️  警告: 无法导入动态语义合并器，将使用原有的静态合并方法")

@dataclass
class ClassificationPath:
    """分类路径数据结构"""
    root: str                    # 根节点 (如: "计算机科学")
    level1: str                  # 一级分类 (如: "计算机视觉") 
    level2: Optional[str]        # 二级分类 (如: "图像处理")
    level3: Optional[str]        # 三级分类 (如: "图像超分辨率")
    depth: int                   # 分类深度 (2-4)
    confidence: float            # 分类置信度
    reasoning: str = ""          # 分类理由
    
    def to_path_string(self) -> str:
        """转换为路径字符串"""
        path_parts = [self.root, self.level1]
        if self.level2:
            path_parts.append(self.level2)
        if self.level3:
            path_parts.append(self.level3)
        return " → ".join(path_parts)
    
    def get_path_list(self) -> List[str]:
        """获取分类路径列表"""
        path_parts = [self.root, self.level1]
        if self.level2:
            path_parts.append(self.level2)
        if self.level3:
            path_parts.append(self.level3)
        return path_parts
    
    def get_leaf_node(self) -> str:
        """获取叶子节点名称"""
        if self.level3:
            return self.level3
        elif self.level2:
            return self.level2
        else:
            return self.level1
    
    def get_parent_node(self) -> str:
        """获取倒数第二层节点名称（用于统计聚合）"""
        if self.depth == 2:
            return self.root  # 2层时，统计聚合到root
        elif self.depth == 3:
            return self.level1  # 3层时，统计聚合到level1
        elif self.depth == 4:
            return self.level2  # 4层时，统计聚合到level2
        return self.root  # 默认返回根节点

class TreeNode:
    """树节点类"""
    def __init__(self, name: str, level: int):
        self.name = name
        self.level = level
        self.children: Dict[str, 'TreeNode'] = {}
        self.papers: List[Dict] = []  # 存储属于该节点的论文
        self.statistics: Dict = {}    # 存储统计信息
        
    def add_child(self, child_name: str) -> 'TreeNode':
        """添加子节点"""
        if child_name not in self.children:
            self.children[child_name] = TreeNode(child_name, self.level + 1)
        return self.children[child_name]
    
    def add_paper(self, paper: Dict):
        """向节点添加论文"""
        self.papers.append(paper)
    
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0
    
    def get_leaf_nodes(self) -> List['TreeNode']:
        """获取所有叶子节点"""
        if self.is_leaf():
            return [self]
        
        leaves = []
        for child in self.children.values():
            leaves.extend(child.get_leaf_nodes())
        return leaves

class TreeClassificationAnalyzer:
    """树形分类分析器"""
    
    def __init__(self, api_key=None, data_dir="../data", ignore_cache=False):
        """初始化分析器"""
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
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
        self.classification_cache_file = Path("cache/classification_cache.pkl")
        self.analysis_cache_file = Path("cache/leaf_analysis_cache.pkl")
        self.classification_cache_file.parent.mkdir(exist_ok=True)
        
        # 数据目录 - 智能检测
        self.data_dir = self._resolve_data_dir(data_dir)
        
        # 加载缓存（如果不忽略的话）
        if ignore_cache:
            logger.info("⚠️ 忽略缓存，将重新分析所有论文")
            self.classification_cache = {}
            self.analysis_cache = {}
        else:
            self.classification_cache = self._load_cache(self.classification_cache_file)
            self.analysis_cache = self._load_cache(self.analysis_cache_file)
        
        # 分类树
        self.classification_tree = TreeNode("根节点", 0)
        
        logger.info("✅ 树形分类分析器初始化成功")
        
        # 第一步：论文分类提示词
        self.classification_prompt = """
你是一名专业的学术分类专家，精通计算机科学、人工智能等技术领域的细分分类。

请对以下论文进行**灵活层次分类**，根据论文内容的具体程度选择合适的分类深度（2-4层）：

**论文标题**: {title}
**论文摘要**: {abstract}
**arXiv类别**: {categories}

**分类层次说明**：
- **2层分类**: 适用于宽泛的跨领域研究 (如: "人工智能" → "机器学习理论")
- **3层分类**: 适用于常规专业研究 (如: "计算机科学" → "计算机视觉" → "图像处理")  
- **4层分类**: 适用于高度专业化研究 (如: "计算机科学" → "计算机视觉" → "图像处理" → "图像超分辨率")

**分类原则**：
1. 根据论文的专业化程度选择分类深度
2. 确保同类论文能够聚集在同一个倒数第二层节点下
3. 避免过度细分导致节点过于稀疏
4. 保持分类的逻辑层次性和语义一致性

**输出格式**：
严格按照以下JSON格式返回，未使用的层级设为null：

{{
  "root": "根节点名称",
  "level1": "一级分类名称", 
  "level2": "二级分类名称或null",
  "level3": "三级分类名称或null",
  "depth": 分类深度数字(2-4),
  "confidence": 0.95,
  "reasoning": "选择该分类深度和路径的理由"
}}

**分类示例**：
- 2层: {{"root": "数学", "level1": "概率论与统计", "level2": null, "level3": null, "depth": 2}}
- 3层: {{"root": "计算机科学", "level1": "计算机视觉", "level2": "图像处理", "level3": null, "depth": 3}}
- 4层: {{"root": "计算机科学", "level1": "计算机视觉", "level2": "图像处理", "level3": "图像超分辨率", "depth": 4}}
"""
        
        # 第二步：叶子节点统计分析提示词
        self.leaf_analysis_prompt = """
你是一名专业的学术统计分析专家，请对以下叶子节点的论文集合进行深度统计分析。

**叶子节点**: {leaf_name}
**分类路径**: {classification_path}
**论文数量**: {paper_count}

**论文摘要集合**:
{abstracts}

**分析要求**：
请对该叶子节点的论文进行以下三个维度的统计分析，每个维度提取TOP10：

1. **关键词分析** (keywords): 提取最频繁出现的技术关键词
2. **技术方法分析** (methods): 总结主要使用的技术方法和算法
3. **研究问题分析** (problems): 识别主要解决的研究问题和挑战

**输出格式**：
严格按照以下JSON格式返回：

{{
  "keywords": [
    {{"term": "关键词1", "frequency": 15, "description": "简短描述"}},
    {{"term": "关键词2", "frequency": 12, "description": "简短描述"}},
    ...
  ],
  "methods": [
    {{"method": "技术方法1", "frequency": 8, "description": "方法描述"}},
    {{"method": "技术方法2", "frequency": 6, "description": "方法描述"}},
    ...
  ],
  "problems": [
    {{"problem": "研究问题1", "frequency": 10, "description": "问题描述"}},
    {{"problem": "研究问题2", "frequency": 7, "description": "问题描述"}},
    ...
  ],
  "summary": "该叶子节点的整体研究趋势和特点总结"
}}

**注意事项**：
- 确保统计结果基于实际论文内容
- 优先提取具有代表性的专业术语
- 频次统计应该准确反映在论文集合中的出现频率
- 每个维度最多返回10个条目，按频次降序排列
"""

    def _resolve_data_dir(self, data_dir: str) -> str:
        """智能解析数据目录路径"""
        # 尝试多个可能的路径
        possible_paths = [
            data_dir,                    # 用户指定的路径
            "../data",                   # 项目根目录下的data
            "../../data",               # 如果在子目录中
            "./data",                    # 当前目录的data
            os.path.join(os.getcwd(), "data"),  # 当前工作目录的data
        ]
        
        for path in possible_paths:
            abs_path = Path(path).resolve()
            if abs_path.exists() and abs_path.is_dir():
                logger.info(f"找到数据目录: {abs_path}")
                return str(abs_path)
        
        # 如果都没找到，返回原始路径（会在后续处理中报错）
        logger.warning(f"未找到数据目录，将使用指定路径: {data_dir}")
        return data_dir
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """加载缓存"""
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"已加载缓存文件: {cache_file}")
                return cache
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
        return {}
    
    def _save_cache(self, cache: Dict, cache_file: Path):
        """保存缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.debug(f"缓存已保存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _convert_old_cache_format(self, cached_result: Dict) -> Dict:
        """
        转换旧格式缓存为新格式
        旧格式: {root, level1, level2, leaf, confidence}
        新格式: {root, level1, level2, level3, depth, confidence, reasoning}
        """
        new_format = {
            'root': cached_result.get('root', '未分类'),
            'level1': cached_result.get('level1', '未知领域'),
            'level2': cached_result.get('level2'),
            'level3': cached_result.get('leaf'),  # 将旧的leaf转为level3
            'depth': 4 if cached_result.get('leaf') else 3,  # 推测深度
            'confidence': cached_result.get('confidence', 0.5),
            'reasoning': '从旧格式缓存转换'
        }
        
        # 处理None值和特殊标记
        if new_format['level2'] in ['未知子领域', None]:
            new_format['level2'] = None
        if new_format['level3'] in ['未知具体领域', None]:
            new_format['level3'] = None
            
        # 重新计算深度
        depth = 2
        if new_format['level2']:
            depth = 3
        if new_format['level3']:
            depth = 4
        new_format['depth'] = depth
        
        logger.debug(f"转换旧格式缓存: {cached_result} -> {new_format}")
        return new_format
    
    def load_papers_by_date(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        根据指定日期加载论文数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            论文数据列表
        """
        papers = []
        
        # 获取数据目录中的所有jsonl文件
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.error(f"数据目录不存在: {data_path.absolute()}")
            logger.error(f"请确保数据目录存在，或使用 --data-dir 参数指定正确的路径")
            return papers
        
        # 收集日期范围内的文件
        target_files = []
        for jsonl_file in data_path.glob("*.jsonl"):
            file_date = jsonl_file.stem  # 获取不带扩展名的文件名
            
            # 检查日期格式和范围
            try:
                file_datetime = datetime.strptime(file_date, "%Y-%m-%d")
                
                # 检查日期范围
                in_range = True
                if start_date:
                    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                    in_range = in_range and file_datetime >= start_datetime
                
                if end_date:
                    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                    in_range = in_range and file_datetime <= end_datetime
                
                if in_range:
                    target_files.append(jsonl_file)
                    
            except ValueError:
                # 跳过日期格式不正确的文件
                continue
        
        # 加载文件内容
        for jsonl_file in sorted(target_files):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            paper = json.loads(line.strip())
                            papers.append(paper)
            except Exception as e:
                logger.warning(f"加载文件失败 {jsonl_file}: {e}")
        
        date_range_str = f"{start_date or '最早'} 到 {end_date or '最新'}"
        logger.info(f"加载论文数据: {len(papers)} 篇 (时间范围: {date_range_str})")
        
        return papers
    
    def is_ai_related_paper(self, title, abstract):
        """
        增强的AI相关性判断，更严格的筛选
        """
        # 组合标题和摘要进行分析
        text = f"{title} {abstract}".lower()
        
        # AI核心关键词 - 大幅扩展和更精准
        ai_core_keywords = {
            # 机器学习与深度学习核心
            'machine learning', 'deep learning', 'neural network', 'neural networks', 
            'artificial intelligence', 'ai', 'deep neural network', 'convolutional neural network',
            'cnn', 'rnn', 'lstm', 'gru', 'transformer', 'attention mechanism',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'semi-supervised learning', 'self-supervised learning', 'contrastive learning',
            'few-shot learning', 'zero-shot learning', 'meta-learning', 'transfer learning',
            'federated learning', 'continual learning', 'lifelong learning',
            
            # 大语言模型与NLP
            'large language model', 'language model', 'llm', 'gpt', 'bert', 'roberta',
            'natural language processing', 'nlp', 'text generation', 'language generation',
            'text-to-text', 'seq2seq', 'encoder-decoder', 'pre-trained model',
            'fine-tuning', 'prompt engineering', 'prompt learning', 'in-context learning',
            'chain-of-thought', 'reasoning', 'question answering', 'dialogue system',
            'conversation', 'chatbot', 'retrieval-augmented generation', 'rag',
            'hallucination', 'text classification', 'sentiment analysis', 'named entity recognition',
            'information extraction', 'machine translation', 'summarization',
            
            # 计算机视觉核心
            'computer vision', 'image processing', 'object detection', 'image classification',
            'semantic segmentation', 'instance segmentation', 'object recognition',
            'face recognition', 'facial recognition', 'pose estimation', 'action recognition',
            'video analysis', 'video understanding', 'image generation', 'text-to-image',
            'diffusion model', 'generative adversarial network', 'gan', 'variational autoencoder',
            'vae', '3d reconstruction', 'depth estimation', 'optical flow',
            'visual tracking', 'multi-object tracking', 'visual question answering',
            'image captioning', 'visual grounding', 'object localization',
            
            # 多模态与视觉-语言
            'multimodal', 'multi-modal', 'vision-language', 'vision language model',
            'clip', 'visual-linguistic', 'cross-modal', 'image-text', 'video-text',
            'audio-visual', 'speech recognition', 'automatic speech recognition',
            
            # AI应用与智能系统
            'autonomous driving', 'self-driving', 'robotics', 'intelligent agent',
            'recommender system', 'recommendation system', 'knowledge graph',
            'graph neural network', 'gnn', 'embedding', 'representation learning',
            'generative model', 'discriminative model', 'adversarial learning',
            'domain adaptation', 'data augmentation', 'active learning',
            
            # AI安全与可解释性
            'adversarial attack', 'adversarial example', 'robustness', 'explainable ai',
            'interpretable ai', 'fairness', 'bias detection', 'model explanation',
            'attribution', 'saliency', 'grad-cam', 'attention visualization'
        }
        
        # 直接排除的非AI领域关键词
        excluded_keywords = {
            # 纯图形学渲染
            'skeletal animation', 'vertex shader', 'fragment shader', 'ray tracing',
            'rasterization', 'procedural generation', 'terrain generation', 'mesh generation',
            'texture mapping', 'normal mapping', 'lighting model', 'brdf', 'specular highlights',
            'real-time rendering', 'gpu rendering', 'opengl', 'directx', 'vulkan',
            
            # 纯网络通信
            'wireless communication', 'channel estimation', 'signal processing',
            'antenna design', 'mimo', 'ofdm', 'network protocol', 'routing protocol',
            'network topology', 'bandwidth allocation', 'terahertz communication',
            'channel modeling', 'path loss', '5g', '6g network', 'cellular network',
            
            # 软件工程
            'uml modeling', 'software architecture', 'design pattern', 'code generation',
            'software testing', 'version control', 'agile development', 'scrum',
            
            # 数字人文
            'digital humanities', 'textual scholarship', 'literary analysis',
            'historical analysis', 'cultural studies', 'philology',
            
            # 纯数学/物理
            'quantum mechanics', 'quantum computing', 'quantum algorithm',
            'mathematical modeling', 'numerical simulation', 'finite element',
            'fluid dynamics', 'thermodynamics', 'electromagnetics',
            
            # 生物医学（非AI辅助）
            'protein folding', 'dna sequencing', 'molecular biology',
            'biochemistry', 'pharmacology', 'clinical trial',
            
            # 其他工程领域
            'mechanical engineering', 'civil engineering', 'electrical circuits',
            'power systems', 'control systems', 'embedded systems'
        }
        
        # 计算AI关键词匹配数
        ai_score = 0
        for keyword in ai_core_keywords:
            if keyword in text:
                ai_score += 1
        
        # 检查排除关键词
        excluded_score = 0
        for keyword in excluded_keywords:
            if keyword in text:
                excluded_score += 1
        
        # 更严格的判断逻辑
        # 必须有至少3个AI关键词匹配，且排除关键词不超过1个
        is_ai_related = (ai_score >= 3) and (excluded_score <= 1)
        
        # 如果AI分数很高(>=5)，即使有少量排除词也接受
        if ai_score >= 5 and excluded_score <= 2:
            is_ai_related = True
            
        # 特殊情况：如果明确包含AI应用但用了传统技术词汇
        ai_applications = ['medical imaging', 'medical image', 'autonomous vehicle', 
                          'intelligent system', 'smart system', 'ai-assisted',
                          'ai-based', 'machine learning-based', 'deep learning-based']
        
        for app in ai_applications:
            if app in text and ai_score >= 2:
                is_ai_related = True
                break
        
        self.logger.debug(f"AI筛选 - 标题: {title[:50]}...")
        self.logger.debug(f"AI分数: {ai_score}, 排除分数: {excluded_score}, 结果: {is_ai_related}")
        
        return is_ai_related

    def classify_paper(self, paper: Dict) -> ClassificationPath:
        """
        第一步：对单篇论文进行细分分类
        
        Args:
            paper: 论文数据
            
        Returns:
            分类路径，如果不是AI相关论文则返回None
        """
        # 首先检查是否为AI相关论文
        if not self.is_ai_related_paper(paper.get('title', ''), paper.get('summary', '')):
            logger.info(f"跳过非AI相关论文: {paper.get('title', '')[:50]}...")
            return None
            
        paper_id = paper.get('id', '')
        
        # 检查缓存
        if paper_id in self.classification_cache:
            cached_result = self.classification_cache[paper_id]
            
            # 处理旧格式缓存兼容性
            if 'leaf' in cached_result:
                # 将旧格式转换为新格式
                cached_result = self._convert_old_cache_format(cached_result)
            
            return ClassificationPath(**cached_result)
        
        # 准备论文信息
        title = paper.get('title', '').strip()
        abstract = paper.get('summary', '').strip()
        categories = paper.get('categories', [])
        
        if not abstract:
            # 返回默认分类
            default_path = ClassificationPath(
                root="未分类",
                level1="未知领域", 
                level2=None,
                level3=None,
                depth=2,
                confidence=0.0
            )
            return default_path
        
        try:
            # 调用大模型进行分类
            prompt = self.classification_prompt.format(
                title=title[:200],  # 限制标题长度
                abstract=abstract[:1500],  # 限制摘要长度
                categories=', '.join(categories[:5])  # 限制类别数量
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名专业的学术分类专家，请严格按照JSON格式返回分类结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析JSON结果
            try:
                # 提取JSON部分
                if '```json' in result_text:
                    json_part = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    json_part = result_text.split('```')[1]
                else:
                    json_part = result_text
                
                result = json.loads(json_part.strip())
                
                # 创建分类路径
                classification_path = ClassificationPath(
                    root=result.get('root', '未分类'),
                    level1=result.get('level1', '未知领域'),
                    level2=result.get('level2'), 
                    level3=result.get('level3'),
                    depth=int(result.get('depth', 2)),
                    confidence=float(result.get('confidence', 0.5)),
                    reasoning=result.get('reasoning', '')
                )
                
                # 保存到缓存
                self.classification_cache[paper_id] = {
                    'root': classification_path.root,
                    'level1': classification_path.level1,
                    'level2': classification_path.level2,
                    'level3': classification_path.level3,
                    'depth': classification_path.depth,
                    'confidence': classification_path.confidence,
                    'reasoning': classification_path.reasoning
                }
                
                return classification_path
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"分类结果解析失败: {e}")
                raise
                
        except Exception as e:
            logger.error(f"论文分类失败 {paper_id}: {e}")
            # 返回默认分类
            return ClassificationPath(
                root="分类失败",
                level1="API错误",
                level2=None,
                level3=None,
                depth=2,
                confidence=0.0
            )
    
    def batch_classify_papers(self, papers: List[Dict], max_workers: int = 8) -> List[Tuple[Dict, ClassificationPath]]:
        """
        批量分类论文
        
        Args:
            papers: 论文列表
            max_workers: 最大并发数
            
        Returns:
            (论文, 分类路径) 元组列表
        """
        results = []
        
        # 分离需要分类的论文和已缓存的论文
        papers_to_classify = []
        cached_results = []
        
        for paper in papers:
            paper_id = paper.get('id', '')
            
            if paper_id in self.classification_cache:
                cached_result = self.classification_cache[paper_id]
                
                # 处理旧格式缓存兼容性
                if 'leaf' in cached_result:
                    cached_result = self._convert_old_cache_format(cached_result)
                
                classification_path = ClassificationPath(**cached_result)
                cached_results.append((paper, classification_path))
            else:
                papers_to_classify.append(paper)
        
        logger.info(f"总论文数: {len(papers)}, 缓存命中: {len(cached_results)}, 需要分类: {len(papers_to_classify)}")
        
        # 处理需要分类的论文
        if papers_to_classify:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交分类任务
                future_to_paper = {
                    executor.submit(self.classify_paper, paper): paper
                    for paper in papers_to_classify
                }
                
                # 收集结果
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        classification_path = future.result()
                        # 如果返回None，说明不是AI相关论文，跳过
                        if classification_path is not None:
                            results.append((paper, classification_path))
                        
                        # 定期保存缓存
                        if len(results) % 50 == 0:
                            self._save_cache(self.classification_cache, self.classification_cache_file)
                            logger.info(f"已完成分类: {len(results) + len(cached_results)}/{len(papers)} (跳过非AI论文)")
                            
                    except Exception as e:
                        logger.error(f"论文分类失败: {e}")
                        # 添加默认分类
                        default_path = ClassificationPath(
                            root="分类失败",
                            level1="处理错误",
                            level2=None,
                            level3=None,
                            depth=2,
                            confidence=0.0
                        )
                        results.append((paper, default_path))
        
        # 合并结果
        all_results = cached_results + results
        
        # 保存缓存
        self._save_cache(self.classification_cache, self.classification_cache_file)
        
        # 统计筛选效果
        total_papers = len(papers)
        ai_papers = len(all_results)
        filtered_out = total_papers - ai_papers
        filter_rate = (filtered_out / total_papers * 100) if total_papers > 0 else 0
        
        logger.info(f"论文分类完成: {ai_papers} 篇AI相关论文 (总计 {total_papers} 篇，筛选掉 {filtered_out} 篇非AI论文，筛选率 {filter_rate:.1f}%)")
        return all_results
    
    def build_classification_tree(self, classified_papers: List[Tuple[Dict, ClassificationPath]]) -> TreeNode:
        """
        构建分类树并将论文分配到叶子节点
        
        Args:
            classified_papers: 已分类的论文列表
            
        Returns:
            分类树根节点
        """
        # 重新初始化分类树
        self.classification_tree = TreeNode("根节点", 0)
        
        # 构建树结构并分配论文
        for paper, classification_path in classified_papers:
            # 添加分类路径到论文数据中
            paper_with_path = paper.copy()
            paper_with_path['classification_path'] = classification_path.to_path_string()
            paper_with_path['classification_confidence'] = classification_path.confidence
            paper_with_path['classification_depth'] = classification_path.depth
            
            # 根据分类深度构建路径
            current_node = self.classification_tree.add_child(classification_path.root)
            current_node = current_node.add_child(classification_path.level1)
            
            if classification_path.level2:
                current_node = current_node.add_child(classification_path.level2)
            
            if classification_path.level3:
                current_node = current_node.add_child(classification_path.level3)
            
            # 将论文添加到叶子节点
            current_node.add_paper(paper_with_path)
        
        logger.info("分类树构建完成")
        return self.classification_tree
    
    def analyze_leaf_node(self, leaf_node: TreeNode) -> Dict:
        """
        第二步：分析单个叶子节点的论文集合
        
        Args:
            leaf_node: 叶子节点
            
        Returns:
            分析结果
        """
        leaf_name = leaf_node.name
        papers = leaf_node.papers
        
        if not papers:
            return {
                'keywords': [],
                'methods': [],
                'problems': [],
                'summary': '该叶子节点没有论文数据'
            }
        
        # 检查缓存
        cache_key = f"{leaf_name}_{len(papers)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # 准备论文摘要
        abstracts = []
        for i, paper in enumerate(papers[:20]):  # 限制最多20篇论文以控制prompt长度
            abstract = paper.get('summary', '').strip()
            if abstract:
                abstracts.append(f"{i+1}. {abstract[:300]}")  # 每篇摘要限制300字符
        
        if not abstracts:
            return {
                'keywords': [],
                'methods': [],
                'problems': [],
                'summary': '该叶子节点的论文缺少摘要信息'
            }
        
        # 构建分类路径信息
        classification_path = "未知"
        if papers:
            first_paper = papers[0]
            classification_path = first_paper.get('classification_path', '未知')
        
        try:
            # 调用大模型进行分析
            prompt = self.leaf_analysis_prompt.format(
                leaf_name=leaf_name,
                classification_path=classification_path,
                paper_count=len(papers),
                abstracts='\n'.join(abstracts)
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名专业的学术统计分析专家，请严格按照JSON格式返回分析结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析JSON结果
            try:
                # 提取JSON部分
                if '```json' in result_text:
                    json_part = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    json_part = result_text.split('```')[1]
                else:
                    json_part = result_text
                
                result = json.loads(json_part.strip())
                
                # 确保每个列表最多10个元素
                result['keywords'] = result.get('keywords', [])[:10]
                result['methods'] = result.get('methods', [])[:10]
                result['problems'] = result.get('problems', [])[:10]
                result['summary'] = result.get('summary', '')
                
                # 保存到缓存
                self.analysis_cache[cache_key] = result
                
                return result
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"叶子节点分析结果解析失败: {e}")
                raise
                
        except Exception as e:
            logger.error(f"叶子节点分析失败 {leaf_name}: {e}")
            return {
                'keywords': [],
                'methods': [],
                'problems': [],
                'summary': f'分析失败: {str(e)}'
            }
    
    def batch_analyze_leaf_nodes(self, max_workers: int = 6) -> Dict:
        """
        批量分析所有叶子节点
        
        Args:
            max_workers: 最大并发数
            
        Returns:
            所有叶子节点的分析结果
        """
        # 获取所有叶子节点
        leaf_nodes = self.classification_tree.get_leaf_nodes()
        
        # 过滤出有论文的叶子节点
        valid_leaf_nodes = [node for node in leaf_nodes if node.papers]
        
        logger.info(f"开始分析叶子节点: {len(valid_leaf_nodes)} 个有效节点")
        
        analysis_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交分析任务
            future_to_node = {
                executor.submit(self.analyze_leaf_node, node): node
                for node in valid_leaf_nodes
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    analysis_result = future.result()
                    analysis_results[node.name] = {
                        'node_info': {
                            'name': node.name,
                            'paper_count': len(node.papers),
                            'classification_path': node.papers[0].get('classification_path', '未知') if node.papers else '未知'
                        },
                        'analysis': analysis_result
                    }
                    
                    completed += 1
                    if completed % 10 == 0 or completed == len(valid_leaf_nodes):
                        logger.info(f"叶子节点分析进度: {completed}/{len(valid_leaf_nodes)}")
                        # 定期保存缓存
                        self._save_cache(self.analysis_cache, self.analysis_cache_file)
                        
                except Exception as e:
                    logger.error(f"叶子节点分析失败 {node.name}: {e}")
        
        # 保存缓存
        self._save_cache(self.analysis_cache, self.analysis_cache_file)
        
        logger.info(f"叶子节点分析完成: {len(analysis_results)} 个节点")
        return analysis_results
    
    def semantic_merge_similar_nodes(self, parent_aggregation: Dict) -> Dict:
        """
        基于语义相似性合并相近的父节点
        """
        # 尝试使用动态语义合并器
        if DynamicSemanticMerger is not None:
            try:
                print("🚀 使用动态语义合并器进行智能合并...")
                merger = DynamicSemanticMerger(similarity_threshold=0.85)
                return merger.merge_similar_nodes(parent_aggregation)
            except Exception as e:
                print(f"⚠️  动态合并失败，回退到静态合并: {e}")
        
        # 回退到原有的静态合并方法
        print("📚 使用静态语义词典进行合并...")
        
        # 扩展的静态语义相似的节点组（包含更多NLP术语）
        semantic_groups = {
            # NLP相关术语 - 大幅扩展
            "大语言模型": [
                "large language models", "large language model", "llm", "llms",
                "大语言模型", "大型语言模型", "语言大模型", "语言模型"
            ],
            "文本生成": [
                "文本生成", "自动文本理解与生成", "文本生成与理解", "文本生成检测",
                "text generation", "natural language generation", "nlg"
            ],
            "对话系统": [
                "对话系统", "对话系统与对话建模", "对话系统与交互式学习",
                "dialogue systems", "dialog systems", "conversational ai"
            ],
            "自然语言处理": [
                "自然语言处理", "natural language processing", "nlp", "computational linguistics"
            ],
            "文本分类": [
                "文本分类", "text classification", "document classification"
            ],
            "机器翻译": [
                "机器翻译", "machine translation", "neural machine translation"
            ],
            "问答系统": [
                "问答系统", "question answering", "qa", "智能问答"
            ],
            "信息抽取": [
                "信息抽取", "information extraction", "命名实体识别", "named entity recognition"
            ],
            "情感分析": [
                "情感分析", "sentiment analysis", "emotion analysis"
            ],
            "多模态学习": [
                "多模态学习", "多模态推理与理解", "视觉问答与多模态理解",
                "multimodal learning", "vision-language"
            ],
            
            # CV相关术语（保持原有）
            "三维视觉与重建": [
                "三维视觉与重建", "三维视觉与场景重建", "三维视觉与几何建模", 
                "三维重建与运动捕捉", "三维重建与场景理解", "三维视觉与场景理解",
                "三维重建与生成", "三维重建与建模", "三维重建与测量", 
                "三维视觉与姿态估计", "三维视觉与图形学", "三维视觉与生成",
                "三维视觉与点云处理", "三维视觉与点云分析", "立体视觉与三维重建",
                "三维重建", "三维重建与视图合成"
            ],
            "医学图像分析": [
                "医学图像分析", "医学影像分析", "医学图像处理"
            ],
            "深度学习": [
                "深度学习", "深度学习可解释性"
            ],
            "目标检测": [
                "目标检测", "目标检测与识别", "目标检测与定位", "目标检测与跟踪", "目标检测与行为分析"
            ],
            "图像生成": [
                "图像生成", "图像生成与编辑", "图像生成与理解", "图像生成与分割"
            ],
            "视觉与语言": [
                "视觉与语言推理", "视觉与语言融合", "视觉-语言模型", "图像描述与视觉语言模型"
            ],
            "机器人感知": [
                "机器人感知与场景理解", "机器人感知与认知", "机器人感知与视觉", "机器人感知与控制"
            ],
            "视频处理": [
                "视频处理", "视频理解", "视频分析与理解", "视频理解与生成"
            ]
        }
        
        merged_aggregation = {}
        used_nodes = set()
        
        # 处理语义分组
        for target_name, similar_nodes in semantic_groups.items():
            found_nodes = []
            total_papers = 0
            total_leaf_count = 0
            all_leaf_nodes = []
            all_keywords = []
            all_methods = []
            all_problems = []
            
            # 查找当前聚合结果中的相似节点
            for node_key in parent_aggregation.keys():
                node_display_name = parent_aggregation[node_key]['node_info'].get('display_name', node_key)
                if node_display_name in similar_nodes:
                    found_nodes.append(node_key)
                    node_data = parent_aggregation[node_key]
                    total_papers += node_data['node_info']['total_papers']
                    total_leaf_count += node_data['node_info']['leaf_count']
                    all_leaf_nodes.extend(node_data['node_info']['leaf_nodes'])
                    all_keywords.extend(node_data['aggregated_analysis']['keywords'])
                    all_methods.extend(node_data['aggregated_analysis']['methods'])
                    all_problems.extend(node_data['aggregated_analysis']['problems'])
                    used_nodes.add(node_key)
            
            # 如果找到多个相似节点，则合并
            if len(found_nodes) > 1:
                print(f"🔗 合并语义相似节点: {[parent_aggregation[node]['node_info'].get('display_name', node) for node in found_nodes]} -> {target_name}")
                
                # 重新聚合统计
                def merge_items(items_list, item_type='keywords'):
                    term_counts = defaultdict(int)
                    term_descriptions = {}
                    
                    for item in items_list:
                        if isinstance(item, dict):
                            if item_type == 'keywords':
                                term = item.get('term', '')
                            elif item_type == 'methods':
                                term = item.get('method', '')
                            elif item_type == 'problems':
                                term = item.get('problem', '')
                            else:
                                term = item.get('term', item.get('method', item.get('problem', '')))
                            
                            frequency = item.get('frequency', 1)
                            description = item.get('description', '')
                            
                            if term:
                                term_counts[term] += frequency
                                if description and term not in term_descriptions:
                                    term_descriptions[term] = description
                    
                    # 排序并取TOP10
                    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    result = []
                    for term, count in sorted_terms:
                        if item_type == 'keywords':
                            result.append({'term': term, 'frequency': count, 'description': term_descriptions.get(term, '')})
                        elif item_type == 'methods':
                            result.append({'method': term, 'frequency': count, 'description': term_descriptions.get(term, '')})
                        elif item_type == 'problems':
                            result.append({'problem': term, 'frequency': count, 'description': term_descriptions.get(term, '')})
                    
                    return result
                
                merged_keywords = merge_items(all_keywords, 'keywords')
                merged_methods = merge_items(all_methods, 'methods')
                merged_problems = merge_items(all_problems, 'problems')
                
                # 生成汇总描述
                summary = f"该领域包含{total_leaf_count}个子领域，共{total_papers}篇论文。"
                if merged_keywords:
                    top_keywords = [item['term'] for item in merged_keywords[:3]]
                    summary += f" 主要关键词：{', '.join(top_keywords)}。"
                
                merged_aggregation[target_name] = {
                    'node_info': {
                        'name': target_name,
                        'display_name': target_name,
                        'total_papers': total_papers,
                        'leaf_count': total_leaf_count,
                        'leaf_nodes': list(set(all_leaf_nodes)),  # 去重
                        'merged_from': [parent_aggregation[node]['node_info'].get('display_name', node) for node in found_nodes]
                    },
                    'aggregated_analysis': {
                        'keywords': merged_keywords,
                        'methods': merged_methods,
                        'problems': merged_problems,
                        'summary': summary
                    }
                }
            elif len(found_nodes) == 1:
                # 单个节点，直接使用目标名称
                node_key = found_nodes[0]
                node_data = parent_aggregation[node_key]
                merged_aggregation[target_name] = {
                    'node_info': {
                        'name': target_name,
                        'display_name': target_name,
                        'total_papers': node_data['node_info']['total_papers'],
                        'leaf_count': node_data['node_info']['leaf_count'],
                        'leaf_nodes': node_data['node_info']['leaf_nodes']
                    },
                    'aggregated_analysis': node_data['aggregated_analysis']
                }
        
        # 添加未被合并的节点
        for node_key, node_data in parent_aggregation.items():
            if node_key not in used_nodes:
                display_name = node_data['node_info'].get('display_name', node_key)
                merged_aggregation[display_name] = {
                    'node_info': {
                        'name': display_name,
                        'display_name': display_name,
                        'total_papers': node_data['node_info']['total_papers'],
                        'leaf_count': node_data['node_info']['leaf_count'],
                        'leaf_nodes': node_data['node_info']['leaf_nodes']
                    },
                    'aggregated_analysis': node_data['aggregated_analysis']
                }
        
        return merged_aggregation

    def aggregate_parent_node_analysis(self, leaf_analysis: Dict) -> Dict:
        """
        将叶子节点分析结果聚合到倒数第二层节点
        
        Args:
            leaf_analysis: 叶子节点分析结果
            
        Returns:
            父节点聚合分析结果
        """
        parent_aggregation = {}
        
        # 按照分类路径聚合论文
        classification_groups = defaultdict(list)
        
        for leaf_name, leaf_result in leaf_analysis.items():
            node_info = leaf_result['node_info']
            classification_path = node_info.get('classification_path', '')
            
            # 解析分类路径找到倒数第二层节点
            path_parts = classification_path.split(' → ')
            
            if len(path_parts) >= 2:
                if len(path_parts) == 2:
                    # 2层：聚合到root
                    parent_node = path_parts[0]
                elif len(path_parts) == 3:
                    # 3层：聚合到level1  
                    parent_node = f"{path_parts[0]} → {path_parts[1]}"
                elif len(path_parts) >= 4:
                    # 4层或更多：聚合到倒数第二层
                    parent_node = " → ".join(path_parts[:-1])
                else:
                    parent_node = "未知父节点"
                
                classification_groups[parent_node].append((leaf_name, leaf_result))
        
        # 对每个父节点进行聚合分析
        for parent_node, leaf_results in classification_groups.items():
            # 统计信息
            total_papers = sum(result[1]['node_info']['paper_count'] for result in leaf_results)
            leaf_count = len(leaf_results)
            
            # 聚合关键词
            all_keywords = []
            all_methods = []
            all_problems = []
            
            for leaf_name, leaf_result in leaf_results:
                analysis = leaf_result['analysis']
                all_keywords.extend(analysis.get('keywords', []))
                all_methods.extend(analysis.get('methods', []))
                all_problems.extend(analysis.get('problems', []))
            
            # 统计频次并取TOP10
            def aggregate_items(items_list, item_type='term'):
                term_counts = defaultdict(int)
                term_descriptions = {}
                
                for item in items_list:
                    if isinstance(item, dict):
                        # 根据item_type提取对应的字段
                        if item_type == 'keywords':
                            term = item.get('term', '')
                        elif item_type == 'methods':
                            term = item.get('method', '')
                        elif item_type == 'problems':
                            term = item.get('problem', '')
                        else:
                            term = item.get('term', item.get('method', item.get('problem', '')))
                        
                        frequency = item.get('frequency', 1)
                        description = item.get('description', '')
                        
                        if term:  # 只处理非空term
                            term_counts[term] += frequency
                            if description and term not in term_descriptions:
                                term_descriptions[term] = description
                
                # 排序并取TOP10
                sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # 根据item_type返回正确的格式
                result = []
                for term, count in sorted_terms:
                    if item_type == 'keywords':
                        result.append({
                            'term': term,
                            'frequency': count,
                            'description': term_descriptions.get(term, '')
                        })
                    elif item_type == 'methods':
                        result.append({
                            'method': term,
                            'frequency': count,
                            'description': term_descriptions.get(term, '')
                        })
                    elif item_type == 'problems':
                        result.append({
                            'problem': term,
                            'frequency': count,
                            'description': term_descriptions.get(term, '')
                        })
                    else:
                        result.append({
                            'term': term,
                            'frequency': count,
                            'description': term_descriptions.get(term, '')
                        })
                
                return result
            
            aggregated_keywords = aggregate_items(all_keywords, 'keywords')
            aggregated_methods = aggregate_items(all_methods, 'methods')  
            aggregated_problems = aggregate_items(all_problems, 'problems')
            
            # 生成汇总描述
            summary = f"该领域包含{leaf_count}个子领域，共{total_papers}篇论文。"
            if aggregated_keywords:
                top_keywords = [item['term'] for item in aggregated_keywords[:3]]
                summary += f" 主要关键词：{', '.join(top_keywords)}。"
            
            # 提取节点的简洁名称（最后一个部分）
            display_name = parent_node.split(' → ')[-1]
            
            parent_aggregation[parent_node] = {
                'node_info': {
                    'name': parent_node,  # 完整路径，用于唯一标识
                    'display_name': display_name,  # 简洁名称，用于显示
                    'total_papers': total_papers,
                    'leaf_count': leaf_count,
                    'leaf_nodes': [result[0] for result in leaf_results]
                },
                'aggregated_analysis': {
                    'keywords': aggregated_keywords,
                    'methods': aggregated_methods,
                    'problems': aggregated_problems,
                    'summary': summary
                }
            }
        
        logger.info(f"父节点聚合分析完成: {len(parent_aggregation)} 个父节点")
        
        # 第二步：基于语义相似性合并相近节点
        logger.info("🔄 开始语义相似性合并...")
        merged_aggregation = self.semantic_merge_similar_nodes(parent_aggregation)
        logger.info(f"✅ 合并完成: {len(parent_aggregation)} -> {len(merged_aggregation)} 个父节点")
        
        return merged_aggregation
    
    def run_tree_analysis(self, start_date: str = None, end_date: str = None, max_workers: int = 6) -> Dict:
        """
        运行完整的树形分析流程
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_workers: 最大并发数
            
        Returns:
            完整的分析结果
        """
        logger.info("🌳 开始树形分类分析...")
        
        # 1. 加载论文数据
        logger.info("📚 第一步：加载论文数据...")
        papers = self.load_papers_by_date(start_date, end_date)
        
        if not papers:
            logger.warning("没有找到论文数据")
            return {'error': '没有找到论文数据'}
        
        # 2. 第一步：论文分类
        logger.info("🏷️ 第二步：论文细分分类...")
        classified_papers = self.batch_classify_papers(papers, max_workers)
        
        # 3. 构建分类树
        logger.info("🌲 第三步：构建分类树...")
        tree_root = self.build_classification_tree(classified_papers)
        
        # 4. 第二步：叶子节点分析
        logger.info("📊 第四步：叶子节点统计分析...")
        leaf_analysis_results = self.batch_analyze_leaf_nodes(max_workers)
        
        # 5. 父节点聚合分析（核心改进）
        logger.info("🔄 第五步：父节点聚合分析...")
        parent_aggregation_results = self.aggregate_parent_node_analysis(leaf_analysis_results)
        
        # 6. 生成总体统计
        logger.info("📈 第六步：生成总体统计...")
        tree_statistics = self.generate_tree_statistics(tree_root, leaf_analysis_results)
        
        # 7. 保存结果
        results = {
            'date_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'total_papers': len(papers),
            'classification_tree_stats': tree_statistics,
            'leaf_analysis_results': leaf_analysis_results,
            'parent_aggregation_results': parent_aggregation_results,  # 新增父节点聚合结果
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # 保存结果到文件
        self.save_analysis_results(results)
        
        logger.info("✅ 树形分类分析完成！")
        return results
    
    def generate_tree_statistics(self, tree_root: TreeNode, leaf_analysis: Dict) -> Dict:
        """生成分类树的统计信息"""
        stats = {
            'total_leaf_nodes': 0,
            'total_papers_classified': 0,
            'classification_distribution': {},
            'top_leaf_nodes_by_papers': [],
            'classification_confidence_stats': {
                'mean': 0.0,
                'std': 0.0,
                'distribution': {}
            }
        }
        
        # 收集所有叶子节点
        leaf_nodes = tree_root.get_leaf_nodes()
        valid_leaf_nodes = [node for node in leaf_nodes if node.papers]
        
        stats['total_leaf_nodes'] = len(valid_leaf_nodes)
        
        # 统计各级分类的分布
        classification_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        confidence_scores = []
        leaf_paper_counts = []
        
        for node in valid_leaf_nodes:
            paper_count = len(node.papers)
            stats['total_papers_classified'] += paper_count
            leaf_paper_counts.append((node.name, paper_count))
            
            # 分析每篇论文的分类路径
            for paper in node.papers:
                classification_path = paper.get('classification_path', '')
                confidence = paper.get('classification_confidence', 0.0)
                confidence_scores.append(confidence)
                
                # 解析分类路径
                if ' → ' in classification_path:
                    path_parts = classification_path.split(' → ')
                    if len(path_parts) >= 4:
                        root, level1, level2, leaf = path_parts[:4]
                        classification_counts[root][level1][level2][leaf] += 1
        
        # 计算置信度统计
        if confidence_scores:
            stats['classification_confidence_stats']['mean'] = float(np.mean(confidence_scores))
            stats['classification_confidence_stats']['std'] = float(np.std(confidence_scores))
            
            # 置信度分布
            confidence_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for low, high in confidence_ranges:
                range_name = f"{low}-{high}"
                count = sum(1 for score in confidence_scores if low <= score < high)
                stats['classification_confidence_stats']['distribution'][range_name] = count
        
        # 转换分类分布为普通字典
        stats['classification_distribution'] = self._convert_nested_defaultdict(classification_counts)
        
        # Top叶子节点排序
        stats['top_leaf_nodes_by_papers'] = sorted(leaf_paper_counts, key=lambda x: x[1], reverse=True)[:20]
        
        return stats
    
    def _convert_nested_defaultdict(self, d):
        """递归转换嵌套的defaultdict为普通dict"""
        if isinstance(d, defaultdict):
            d = dict(d)
        
        for key, value in d.items():
            if isinstance(value, defaultdict):
                d[key] = self._convert_nested_defaultdict(value)
        
        return d
    
    def save_analysis_results(self, results: Dict, output_dir: str = "tree_analysis_results"):
        """保存分析结果"""
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存完整结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 完整结果文件
        full_results_file = output_path / f"tree_analysis_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 叶子节点分析摘要
        summary_file = output_path / f"leaf_analysis_summary_{timestamp}.json"
        leaf_summary = {}
        for leaf_name, leaf_data in results.get('leaf_analysis_results', {}).items():
            leaf_summary[leaf_name] = {
                'paper_count': leaf_data['node_info']['paper_count'],
                'classification_path': leaf_data['node_info']['classification_path'],
                'top_keywords': leaf_data['analysis']['keywords'][:5],
                'top_methods': leaf_data['analysis']['methods'][:5],
                'top_problems': leaf_data['analysis']['problems'][:5],
                'summary': leaf_data['analysis']['summary'][:200] + '...' if len(leaf_data['analysis']['summary']) > 200 else leaf_data['analysis']['summary']
            }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(leaf_summary, f, ensure_ascii=False, indent=2)
        
        # 父节点聚合结果摘要
        parent_summary_file = output_path / f"parent_aggregation_summary_{timestamp}.json"
        parent_summary = {}
        for parent_name, parent_data in results.get('parent_aggregation_results', {}).items():
            display_name = parent_data['node_info'].get('display_name', parent_name)
            parent_summary[display_name] = {
                'full_path': parent_name,
                'total_papers': parent_data['node_info']['total_papers'],
                'leaf_count': parent_data['node_info']['leaf_count'],
                'leaf_nodes': parent_data['node_info']['leaf_nodes'],
                'top_keywords': parent_data['aggregated_analysis']['keywords'][:10],
                'top_methods': parent_data['aggregated_analysis']['methods'][:10],
                'top_problems': parent_data['aggregated_analysis']['problems'][:10],
                'summary': parent_data['aggregated_analysis']['summary']
            }
        
        with open(parent_summary_file, 'w', encoding='utf-8') as f:
            json.dump(parent_summary, f, ensure_ascii=False, indent=2)
        
        # CSV格式的叶子节点统计
        csv_file = output_path / f"leaf_nodes_stats_{timestamp}.csv"
        csv_data = []
        for leaf_name, leaf_data in results.get('leaf_analysis_results', {}).items():
            csv_data.append({
                'leaf_node': leaf_name,
                'classification_path': leaf_data['node_info']['classification_path'],
                'paper_count': leaf_data['node_info']['paper_count'],
                'top_keywords': '; '.join([item['term'] for item in leaf_data['analysis']['keywords'][:5]]),
                'top_methods': '; '.join([item['method'] for item in leaf_data['analysis']['methods'][:5]]),
                'top_problems': '; '.join([item['problem'] for item in leaf_data['analysis']['problems'][:5]])
            })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # CSV格式的父节点聚合统计
        parent_csv_file = output_path / f"parent_aggregation_stats_{timestamp}.csv"
        parent_csv_data = []
        for parent_name, parent_data in results.get('parent_aggregation_results', {}).items():
            display_name = parent_data['node_info'].get('display_name', parent_name)
            parent_csv_data.append({
                'parent_node': display_name,
                'full_path': parent_name,
                'total_papers': parent_data['node_info']['total_papers'],
                'leaf_count': parent_data['node_info']['leaf_count'],
                'top_keywords': '; '.join([item.get('term', '') for item in parent_data['aggregated_analysis']['keywords'][:5]]),
                'top_methods': '; '.join([item.get('method', '') for item in parent_data['aggregated_analysis']['methods'][:5]]),
                'top_problems': '; '.join([item.get('problem', '') for item in parent_data['aggregated_analysis']['problems'][:5]]),
                'summary': parent_data['aggregated_analysis']['summary']
            })
        
        if parent_csv_data:
            parent_df = pd.DataFrame(parent_csv_data)
            parent_df.to_csv(parent_csv_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"分析结果已保存到: {output_path}")
        logger.info(f"  完整结果: {full_results_file}")
        logger.info(f"  叶子节点摘要: {summary_file}")
        logger.info(f"  父节点聚合摘要: {parent_summary_file}")
        logger.info(f"  叶子节点CSV: {csv_file}")
        logger.info(f"  父节点聚合CSV: {parent_csv_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='论文树形分类分析系统')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--api-key', help='OpenAI API Key')
    parser.add_argument('--data-dir', default='data', help='数据目录')
    parser.add_argument('--max-workers', type=int, default=6, help='最大并发数')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = TreeClassificationAnalyzer(
            api_key=args.api_key,
            data_dir=args.data_dir
        )
        
        # 运行分析
        results = analyzer.run_tree_analysis(
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=args.max_workers
        )
        
        if 'error' in results:
            print(f"❌ 分析失败: {results['error']}")
            return
        
        # 显示结果摘要
        print("\n🎉 分析完成！")
        print(f"📊 总论文数: {results['total_papers']}")
        print(f"🌲 叶子节点数: {results['classification_tree_stats']['total_leaf_nodes']}")
        print(f"📚 已分类论文: {results['classification_tree_stats']['total_papers_classified']}")
        print(f"🔄 父节点聚合: {len(results['parent_aggregation_results'])} 个父节点")
        
        # 显示Top父节点聚合（核心改进效果）
        print("\n📈 论文数量最多的父节点聚合:")
        parent_nodes = [(name, data['node_info']['total_papers']) 
                       for name, data in results['parent_aggregation_results'].items()]
        parent_nodes.sort(key=lambda x: x[1], reverse=True)
        
        for i, (node_name, paper_count) in enumerate(parent_nodes[:10], 1):
            parent_data = results['parent_aggregation_results'][node_name]
            leaf_count = parent_data['node_info']['leaf_count']
            display_name = parent_data['node_info'].get('display_name', node_name)
            print(f"  {i:2d}. {display_name}: {paper_count} 篇论文 ({leaf_count} 个子领域)")
        
        # 显示Top叶子节点（对比）
        print("\n📊 论文数量最多的叶子节点（对比）:")
        top_nodes = results['classification_tree_stats']['top_leaf_nodes_by_papers'][:5]
        for i, (node_name, paper_count) in enumerate(top_nodes, 1):
            print(f"  {i:2d}. {node_name}: {paper_count} 篇")
        
        print(f"\n💾 详细结果已保存到 tree_analysis_results/ 目录")
        print("📋 核心改进：现在统计聚合到倒数第二层节点，有效解决了叶子节点过多的问题！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")

if __name__ == "__main__":
    main()