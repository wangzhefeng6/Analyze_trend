#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动态语义相似性合并模块
基于词向量相似度自动识别和合并语义相近的节点
"""

import os
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import re
import jieba
import openai
from dataclasses import dataclass

@dataclass
class SimilarityCluster:
    """相似性聚类结果"""
    main_term: str
    similar_terms: List[str]
    similarity_scores: List[float]
    merged_papers: int
    merged_leaves: int

class DynamicSemanticMerger:
    """动态语义相似性合并器"""
    
    def __init__(self, api_key: str = None, similarity_threshold: float = 0.85):
        """
        初始化语义合并器
        
        Args:
            api_key: OpenAI API密钥（用于embedding）
            similarity_threshold: 相似度阈值，超过此值的节点会被合并
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
        # 初始化OpenAI客户端
        if self.api_key:
            openai.api_key = self.api_key
        
        # 缓存向量以避免重复计算
        self.embedding_cache = {}
        
        # 预定义的同义词规则（作为补充）
        self.synonym_rules = {
            # NLP相关 - 新增大量NLP术语
            'large_language_models': [
                'large language models', 'large language model', 'llm', 'llms',
                '大语言模型', '大型语言模型', '语言大模型', '语言模型'
            ],
            'text_generation': [
                'text generation', 'natural language generation',
                'nlg', '文本生成', '自然语言生成', '自动文本理解与生成'
            ],
            'dialogue_systems': [
                'dialogue systems', 'dialog systems', 'conversational ai',
                'chatbot', 'conversation', '对话系统', '会话系统', '对话系统与对话建模'
            ],
            'machine_translation': [
                'machine translation', 'neural machine translation', 'nmt',
                '机器翻译', '神经机器翻译'
            ],
            'question_answering': [
                'question answering', 'qa', 'question-answering',
                '问答系统', '问答', '智能问答'
            ],
            'information_extraction': [
                'information extraction', 'ie', 'named entity recognition',
                'ner', 'relation extraction', '信息抽取', '命名实体识别'
            ],
            'sentiment_analysis': [
                'sentiment analysis', 'emotion analysis', 'opinion mining',
                '情感分析', '情绪分析', '意见挖掘'
            ],
            'natural_language_processing': [
                'natural language processing', 'nlp', 'computational linguistics',
                '自然语言处理', '计算语言学'
            ],
            'text_classification': [
                'text classification', 'document classification', 'text categorization',
                '文本分类', '文档分类'
            ],
            'multimodal_learning': [
                'multimodal learning', 'multi-modal learning', 'vision-language',
                '多模态学习', '多模态推理与理解', '视觉问答与多模态理解', '视觉与语言'
            ],
            
            # CV相关（保持原有逻辑）
            'three_d_vision': [
                '三维视觉与重建', '三维视觉与场景重建', '三维视觉与几何建模',
                '三维重建与运动捕捉', '三维重建与场景理解', '三维视觉与场景理解',
                '立体视觉与三维重建', '三维重建', '3d reconstruction', '3d vision'
            ],
            'medical_imaging': [
                '医学图像分析', '医学影像分析', '医学图像处理',
                'medical imaging', 'medical image analysis'
            ],
            'object_detection': [
                '目标检测', '目标检测与识别', '目标检测与定位',
                'object detection', 'object recognition', '目标检测与跟踪'
            ],
            'image_generation': [
                '图像生成', '图像生成与编辑', '图像合成', '图像生成与理解',
                'image generation', 'image synthesis'
            ],
            'video_analysis': [
                '视频处理', '视频理解', '视频分析与理解', '视频理解与生成',
                'video processing', 'video understanding'
            ],
            
            # 机器学习相关
            'deep_learning': [
                'deep learning', '深度学习', 'neural networks', '神经网络', '深度学习可解释性'
            ],
            'machine_learning': [
                'machine learning', '机器学习', 'ml', 'artificial intelligence', 'ai'
            ],
            'reinforcement_learning': [
                'reinforcement learning', 'rl', '强化学习', '增强学习'
            ],
            
            # 机器人相关
            'robotics_perception': [
                '机器人感知与场景理解', '机器人感知与认知', '机器人感知与视觉', 
                '机器人感知与控制', 'robotics perception', 'robot vision'
            ]
        }
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            文本向量
        """
        # 检查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # 使用OpenAI Embedding API
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response['data'][0]['embedding'])
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            self.logger.warning(f"获取embedding失败，使用简单方法: {e}")
            # 降级到简单的字符级相似度
            return self._simple_text_vector(text)
    
    def _simple_text_vector(self, text: str) -> np.ndarray:
        """
        简单的文本向量化方法（降级方案）
        
        Args:
            text: 输入文本
            
        Returns:
            简单向量表示
        """
        # 清理文本
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # 分词（中英文）
        words = []
        # 中文分词
        if re.search(r'[\u4e00-\u9fff]', text):
            words.extend(jieba.cut(text))
        else:
            words.extend(text.split())
        
        # 创建词频向量（简单实现）
        vocab = set()
        for word in words:
            vocab.add(word.strip())
        
        vocab = sorted(list(vocab))
        vector = np.zeros(max(len(vocab), 10))  # 最少10维
        
        for i, word in enumerate(vocab):
            if i < len(vector):
                vector[i] = text.count(word)
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def check_synonym_rules(self, terms: List[str]) -> Dict[str, List[str]]:
        """
        基于预定义规则检查同义词
        
        Args:
            terms: 待检查的术语列表
            
        Returns:
            同义词分组结果
        """
        groups = {}
        used_terms = set()
        
        for rule_name, synonyms in self.synonym_rules.items():
            matched_terms = []
            
            for term in terms:
                if term in used_terms:
                    continue
                    
                # 检查是否匹配同义词规则
                term_lower = term.lower().strip()
                for synonym in synonyms:
                    synonym_lower = synonym.lower().strip()
                    # 更宽松的匹配规则
                    if (term_lower == synonym_lower or 
                        synonym_lower in term_lower or
                        term_lower in synonym_lower or
                        self._fuzzy_match(term_lower, synonym_lower)):
                        matched_terms.append(term)
                        used_terms.add(term)
                        break
            
            if len(matched_terms) > 1:
                # 选择最常见或最短的作为主术语
                main_term = min(matched_terms, key=len)
                groups[main_term] = matched_terms
        
        return groups
    
    def _fuzzy_match(self, term1: str, term2: str) -> bool:
        """
        模糊匹配两个术语
        """
        # 去除常见词汇
        stop_words = {'and', 'or', 'the', 'a', 'an', 'with', 'for', 'to', 'of', 'in'}
        
        words1 = set(term1.split()) - stop_words
        words2 = set(term2.split()) - stop_words
        
        if not words1 or not words2:
            return False
        
        # 计算词汇重叠度
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio > 0.6  # 60%以上重叠认为是相似的
    
    def compute_similarity_matrix(self, terms: List[str]) -> np.ndarray:
        """
        计算术语相似度矩阵
        
        Args:
            terms: 术语列表
            
        Returns:
            相似度矩阵
        """
        embeddings = []
        for term in terms:
            embedding = self.get_text_embedding(term)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def cluster_similar_terms(self, terms: List[str], node_data: Dict[str, Dict]) -> List[SimilarityCluster]:
        """
        聚类相似术语
        
        Args:
            terms: 术语列表
            node_data: 节点数据字典
            
        Returns:
            相似性聚类结果
        """
        if len(terms) < 2:
            return []
        
        # 1. 首先使用预定义规则
        rule_groups = self.check_synonym_rules(terms)
        clusters = []
        used_terms = set()
        
        for main_term, similar_terms in rule_groups.items():
            total_papers = sum(node_data.get(term, {}).get('node_info', {}).get('total_papers', 0) for term in similar_terms)
            total_leaves = sum(node_data.get(term, {}).get('node_info', {}).get('leaf_count', 0) for term in similar_terms)
            
            clusters.append(SimilarityCluster(
                main_term=main_term,
                similar_terms=similar_terms,
                similarity_scores=[1.0] * len(similar_terms),  # 规则匹配给最高分
                merged_papers=total_papers,
                merged_leaves=total_leaves
            ))
            used_terms.update(similar_terms)
        
        # 2. 对未使用规则匹配的术语进行向量聚类
        remaining_terms = [term for term in terms if term not in used_terms]
        
        if len(remaining_terms) >= 2:
            try:
                similarity_matrix = self.compute_similarity_matrix(remaining_terms)
                
                # 使用层次聚类
                distance_matrix = 1 - similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - self.similarity_threshold,
                    affinity='precomputed',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(distance_matrix)
                
                # 组织聚类结果
                cluster_groups = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    cluster_groups[label].append(remaining_terms[i])
                
                # 转换为SimilarityCluster对象
                for cluster_id, cluster_terms in cluster_groups.items():
                    if len(cluster_terms) > 1:
                        # 选择论文数最多的作为主术语
                        main_term = max(cluster_terms, 
                                      key=lambda t: node_data.get(t, {}).get('node_info', {}).get('total_papers', 0))
                        
                        # 计算相似度分数
                        main_idx = remaining_terms.index(main_term)
                        similarity_scores = []
                        for term in cluster_terms:
                            term_idx = remaining_terms.index(term)
                            similarity_scores.append(similarity_matrix[main_idx][term_idx])
                        
                        total_papers = sum(node_data.get(term, {}).get('node_info', {}).get('total_papers', 0) for term in cluster_terms)
                        total_leaves = sum(node_data.get(term, {}).get('node_info', {}).get('leaf_count', 0) for term in cluster_terms)
                        
                        clusters.append(SimilarityCluster(
                            main_term=main_term,
                            similar_terms=cluster_terms,
                            similarity_scores=similarity_scores,
                            merged_papers=total_papers,
                            merged_leaves=total_leaves
                        ))
                        
            except Exception as e:
                self.logger.warning(f"向量聚类失败: {e}")
        
        return clusters
    
    def merge_similar_nodes(self, parent_aggregation: Dict) -> Dict:
        """
        动态合并语义相似的节点
        
        Args:
            parent_aggregation: 父节点聚合结果
            
        Returns:
            合并后的结果
        """
        # 提取所有节点的显示名称
        display_names = []
        name_to_key = {}
        
        for key, data in parent_aggregation.items():
            display_name = data['node_info'].get('display_name', key)
            display_names.append(display_name)
            name_to_key[display_name] = key
        
        print(f"🔍 开始动态语义相似性分析...")
        print(f"   待分析节点数: {len(display_names)}")
        print(f"   相似度阈值: {self.similarity_threshold}")
        
        # 使用预定义规则进行分组
        rule_groups = self.check_synonym_rules(display_names)
        
        merged_result = {}
        used_keys = set()
        
        print(f"🔗 发现 {len(rule_groups)} 个相似性聚类:")
        
        # 处理规则匹配的分组
        for i, (main_term, similar_terms) in enumerate(rule_groups.items(), 1):
            print(f"   {i}. {main_term} (合并 {len(similar_terms)} 个节点)")
            print(f"      └─ 包含: {', '.join(similar_terms)}")
            
            # 合并节点数据
            all_keywords = []
            all_methods = []
            all_problems = []
            all_leaf_nodes = []
            total_papers = 0
            total_leaf_count = 0
            
            for term in similar_terms:
                key = name_to_key[term]
                node_data = parent_aggregation[key]
                
                total_papers += node_data['node_info']['total_papers']
                total_leaf_count += node_data['node_info']['leaf_count']
                all_leaf_nodes.extend(node_data['node_info']['leaf_nodes'])
                all_keywords.extend(node_data['aggregated_analysis']['keywords'])
                all_methods.extend(node_data['aggregated_analysis']['methods'])
                all_problems.extend(node_data['aggregated_analysis']['problems'])
                
                used_keys.add(key)
            
            print(f"      └─ 论文数: {total_papers}, 子领域: {total_leaf_count}")
            
            # 重新聚合统计数据
            merged_keywords = self._merge_frequency_items(all_keywords, 'term')
            merged_methods = self._merge_frequency_items(all_methods, 'method')
            merged_problems = self._merge_frequency_items(all_problems, 'problem')
            
            # 生成摘要
            summary = f"该领域包含{total_leaf_count}个子领域，共{total_papers}篇论文。"
            if merged_keywords:
                top_keywords = [item['term'] for item in merged_keywords[:3]]
                summary += f" 主要关键词：{', '.join(top_keywords)}。"
            
            merged_result[main_term] = {
                'node_info': {
                    'name': main_term,
                    'display_name': main_term,
                    'total_papers': total_papers,
                    'leaf_count': total_leaf_count,
                    'leaf_nodes': list(set(all_leaf_nodes)),
                    'merged_from': similar_terms
                },
                'aggregated_analysis': {
                    'keywords': merged_keywords,
                    'methods': merged_methods,
                    'problems': merged_problems,
                    'summary': summary
                }
            }
        
        # 添加未被合并的节点
        for key, data in parent_aggregation.items():
            if key not in used_keys:
                display_name = data['node_info'].get('display_name', key)
                merged_result[display_name] = {
                    'node_info': {
                        'name': display_name,
                        'display_name': display_name,
                        'total_papers': data['node_info']['total_papers'],
                        'leaf_count': data['node_info']['leaf_count'],
                        'leaf_nodes': data['node_info']['leaf_nodes']
                    },
                    'aggregated_analysis': data['aggregated_analysis']
                }
        
        print(f"\n✅ 合并完成: {len(parent_aggregation)} -> {len(merged_result)} 个节点")
        return merged_result
    
    def _merge_frequency_items(self, items_list: List[Dict], key_field: str) -> List[Dict]:
        """
        合并频次统计项目
        
        Args:
            items_list: 项目列表
            key_field: 关键字段名 ('term', 'method', 'problem')
            
        Returns:
            合并后的TOP10项目
        """
        term_counts = defaultdict(int)
        term_descriptions = {}
        
        for item in items_list:
            if isinstance(item, dict):
                term = item.get(key_field, '')
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
            result.append({
                key_field: term,
                'frequency': count,
                'description': term_descriptions.get(term, '')
            })
        
        return result

def test_dynamic_merger():
    """测试动态语义合并器"""
    
    # 模拟包含NLP和CV节点的数据
    mock_data = {
        "text_generation": {
            'node_info': {
                'display_name': "文本生成",
                'total_papers': 74,
                'leaf_count': 20,
                'leaf_nodes': ["GPT生成", "文本摘要"]
            },
            'aggregated_analysis': {
                'keywords': [{'term': 'Large Language Models', 'frequency': 30}],
                'methods': [{'method': 'Transformer', 'frequency': 25}],
                'problems': [{'problem': 'hallucination', 'frequency': 15}],
                'summary': '文本生成相关研究'
            }
        },
        "large_language_models": {
            'node_info': {
                'display_name': "大语言模型",
                'total_papers': 43,
                'leaf_count': 15,
                'leaf_nodes': ["LLM训练", "模型对齐"]
            },
            'aggregated_analysis': {
                'keywords': [{'term': 'LLM', 'frequency': 40}],
                'methods': [{'method': 'fine-tuning', 'frequency': 30}],
                'problems': [{'problem': 'bias', 'frequency': 20}],
                'summary': '大语言模型研究'
            }
        },
        "natural_language_processing": {
            'node_info': {
                'display_name': "自然语言处理",
                'total_papers': 25,
                'leaf_count': 10,
                'leaf_nodes': ["NER", "情感分析"]
            },
            'aggregated_analysis': {
                'keywords': [{'term': 'NLP', 'frequency': 25}],
                'methods': [{'method': 'BERT', 'frequency': 20}],
                'problems': [{'problem': 'low resource', 'frequency': 10}],
                'summary': 'NLP相关研究'
            }
        },
        "medical_image_analysis": {
            'node_info': {
                'display_name': "医学图像分析",
                'total_papers': 35,
                'leaf_count': 8,
                'leaf_nodes': ["CT分析", "MRI处理"]
            },
            'aggregated_analysis': {
                'keywords': [{'term': 'medical imaging', 'frequency': 30}],
                'methods': [{'method': 'CNN', 'frequency': 25}],
                'problems': [{'problem': 'data scarcity', 'frequency': 15}],
                'summary': '医学图像分析'
            }
        },
        "medical_imaging": {
            'node_info': {
                'display_name': "医学影像分析",
                'total_papers': 12,
                'leaf_count': 3,
                'leaf_nodes': ["影像诊断"]
            },
            'aggregated_analysis': {
                'keywords': [{'term': 'medical image', 'frequency': 12}],
                'methods': [{'method': 'deep learning', 'frequency': 10}],
                'problems': [{'problem': 'annotation cost', 'frequency': 8}],
                'summary': '医学影像相关'
            }
        }
    }
    
    # 测试动态合并
    merger = DynamicSemanticMerger(similarity_threshold=0.8)
    result = merger.merge_similar_nodes(mock_data)
    
    print(f"\n📊 测试结果:")
    for name, data in result.items():
        info = data['node_info']
        print(f"  ✓ {name}: {info['total_papers']} 篇论文 ({info['leaf_count']} 子领域)")
        if 'merged_from' in info:
            print(f"    └─ 合并自: {', '.join(info['merged_from'])}")

if __name__ == "__main__":
    test_dynamic_merger() 