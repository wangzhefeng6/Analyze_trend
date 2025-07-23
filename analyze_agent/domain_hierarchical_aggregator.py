import os
import json
import glob
from collections import defaultdict, Counter
from typing import List, Dict, Any
import argparse
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入进度条库
try:
    from tqdm import tqdm
except ImportError:
    print("[Warning] tqdm包未安装，将使用简单进度显示。请先 pip install tqdm")
    # 创建一个简单的进度条替代品
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or len(iterable) if iterable else 0
            self.desc = desc or ""
            self.current = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            print(f"\n{self.desc} 完成!")
            
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.current % 10 == 0:  # 每10个显示一次进度
                    print(f"\r{self.desc} 进度: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)", end="", flush=True)
                yield item
                
        def update(self, n=1):
            self.current += n
            if self.current % 10 == 0:  # 每10个显示一次进度
                print(f"\r{self.desc} 进度: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)", end="", flush=True)

DATA_DIR = '../data'
OUTPUT_DIR = './output'
CACHE_FILE = './classification_cache.json'

# ========== OpenAI API集成（新版用法） ==========
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[Warning] openai包未安装，分类功能不可用。请先 pip install openai")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY','sk-ngUapJAt1q3Dm5d2094aDe236d60422589B9Ab4bAaF933Ab')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.dou.chat/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'openai/gpt-4.1')

# 加载/保存分类缓存
def load_classification_cache(cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_classification_cache(cache, cache_file=CACHE_FILE):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 调用OpenAI API对单篇论文分类（新版client方式）
def classify_paper_with_openai(paper, cache, client, model=OPENAI_MODEL):
    paper_id = paper.get('id') or paper.get('title')
    if not paper_id:
        return None
    if paper_id in cache and cache[paper_id].get('classification_path') and all(
        k in cache[paper_id] for k in ['keywords', 'methods', 'problems']
    ):
        paper['keywords'] = cache[paper_id]['keywords']
        paper['methods'] = cache[paper_id]['methods']
        paper['problems'] = cache[paper_id]['problems']
        # 后处理归一化
        paper['classification_path'] = normalize_classification_path(cache[paper_id]['classification_path'])
        return paper['classification_path']
    if not client:
        print("[Error] openai包未安装或API Key未设置，无法分类。")
        return None

    # 修正：强制检查title和abstract/summary
    title = (paper.get('title') or '').strip()
    abstract = (paper.get('abstract') or '').strip() or (paper.get('summary') or '').strip()
    
    # 详细检查数据完整性
    if not title and not abstract:
        print(f"[Data Error] 论文 {paper_id} 缺少标题和摘要，跳过。")
        return None
    elif not title:
        print(f"[Data Error] 论文 {paper_id} 缺少标题，跳过。")
        return None
    elif not abstract:
        print(f"[Data Warning] 论文 {paper_id} 缺少摘要，仅使用标题进行分类。")
        # 继续处理，但记录警告

    categories = paper.get('categories', [])
    
    # 构建论文内容
    paper_content = f"标题: {title}\n"
    if abstract:
        paper_content += f"摘要: {abstract}\n"
    if categories:
        paper_content += f"分类: {', '.join(categories)}"
    
    prompt = f"""
你是一名专业的学术分类与信息抽取专家，精通计算机科学、人工智能等技术领域的细分分类。
请对以下论文进行**分层分类**，并抽取其关键词、主要方法、研究问题，严格按照JSON格式输出。

**论文内容：**
{paper_content}

**分类要求：**
- 必须严格按照下方标准领域词表选择每一级的分类名称，不允许自由组合或用“与”“和”等连接词。
- 每一级只能选一个标准名称，不能用“人工智能”作为二级领域。
- 如果论文属于交叉领域，请根据主要研究内容选择最合适的二级领域。
- **层级严格限制为最多4级（root, level1, level2, level3），超过4级请截断到第4级。**
- **第4级（level3）只允许出现具体任务/方法/应用，不要包含技术细节、模型名称、算法细节等。**
- 未使用的层级请设为null。
- **绝不允许任何一级分类为“其它”或“其他”，如无法归类请根据论文内容强行选择最接近的标准领域或任务，不能输出“其它”或“其他”。**
- 如果实在无法归类，请选择你认为最接近的标准领域，并在reasoning中说明理由。

**错误示例（不要这样输出）：**
- "root → 计算机视觉 → 其它"
- "root → 其它 → 其它"
- "root → 计算机视觉 → 生成模型 → 其它"

**正确示例：**
- "root → 计算机视觉 → 生成模型 → 扩散模型"
- "root → 计算机视觉 → 目标检测 → 小目标检测"
- "root → 计算机视觉 → 三维重建 → 人体姿态估计"

**标准领域词表：**
- 一级领域（root）：["计算机科学"]
- 二级领域（level1）：["计算机视觉", "自然语言处理", "语音与音频处理", "机器学习", "强化学习", "多模态学习", "机器人学", "人机交互", "医学图像", "大语言模型", "数据可视化"]
- 三级领域（level2）：["目标检测", "图像分割", "生成模型", "问答系统", "对话系统", "语音识别", "视频理解", "代码生成", "医学图像分割", "小样本学习", "三维重建"]
- 四级领域（level3）：如"医学图像分割"、"小样本学习"、"多目标强化学习"等具体任务/方法/应用，或null

**输出格式**（严格JSON）：
{{
  "root": "一级领域（必须为'计算机科学'）",
  "level1": "二级领域（标准词表中选一）",
  "level2": "三级领域（标准词表中选一）",
  "level3": "四级领域（具体任务/方法/应用或null）",
  "depth": 分类深度数字(2-4),
  "confidence": 0.95,
  "reasoning": "选择该分类路径的理由",
  "keywords": ["关键词1", "关键词2", ...],
  "methods": ["方法1", "方法2", ...],
  "problems": ["研究问题1", "研究问题2", ...]
}}
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一名专业的学术分类专家，请严格按照JSON格式返回分类结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=700
        )
        result_text = completion.choices[0].message.content.strip()
        import re
        import ast
        json_part = result_text
        if '```json' in result_text:
            json_part = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            json_part = result_text.split('```')[1]
        try:
            result = json.loads(json_part)
        except Exception as json_error:
            try:
                result = ast.literal_eval(json_part)
            except Exception as ast_error:
                # 检查是否是LLM返回的"请提供内容"类错误
                if any(keyword in result_text for keyword in ["请提供", "需要分类", "论文标题", "论文内容"]):
                    print(f"[LLM Error] 论文 {paper_id} LLM返回内容缺失提示: {result_text[:100]}...")
                    return None
                else:
                    print(f"[Parse Error] 论文 {paper_id} JSON解析失败: {result_text[:100]}...")
                    print(f"  JSON错误: {json_error}")
                    print(f"  AST错误: {ast_error}")
                    return None
        path_parts = [result.get('root'), result.get('level1')]
        if result.get('level2'):
            path_parts.append(result['level2'])
        if result.get('level3'):
            path_parts.append(result['level3'])
        path = ' → '.join([p for p in path_parts if p])
        # 领域路径归一化
        norm_path = normalize_classification_path(path)
        paper['keywords'] = result.get('keywords', [])
        paper['methods'] = result.get('methods', [])
        paper['problems'] = result.get('problems', [])
        cache[paper_id] = {
            'classification_path': norm_path,
            'raw_result': result,
            'keywords': paper['keywords'],
            'methods': paper['methods'],
            'problems': paper['problems']
        }
        return norm_path
    except Exception as e:
        # 区分不同类型的API错误
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            print(f"[API Error] 论文 {paper_id} API配额限制: {error_msg}")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print(f"[API Error] 论文 {paper_id} API超时: {error_msg}")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print(f"[API Error] 论文 {paper_id} API认证失败: {error_msg}")
        elif "model" in error_msg.lower() or "not found" in error_msg.lower():
            print(f"[API Error] 论文 {paper_id} 模型不存在: {error_msg}")
        else:
            print(f"[API Error] 论文 {paper_id} OpenAI API调用失败: {error_msg}")
        return None


def check_paper_data_integrity(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    """检查论文数据完整性，返回统计信息"""
    stats = {
        'total': len(papers),
        'missing_title': 0,
        'missing_abstract': 0,
        'missing_both': 0,
        'empty_title': 0,
        'empty_abstract': 0,
        'valid': 0
    }
    
    # 使用进度条显示数据完整性检查进度
    with tqdm(total=len(papers), desc="检查数据完整性", unit="篇") as pbar:
        for paper in papers:
            paper_id = paper.get('id') or paper.get('title', 'unknown')
            title = (paper.get('title') or '').strip()
            abstract = (paper.get('abstract') or '').strip() or (paper.get('summary') or '').strip()
            
            if not title and not abstract:
                stats['missing_both'] += 1
            elif not title:
                stats['missing_title'] += 1
            elif not abstract:
                stats['missing_abstract'] += 1
            elif title == '':
                stats['empty_title'] += 1
            elif abstract == '':
                stats['empty_abstract'] += 1
            else:
                stats['valid'] += 1
            
            pbar.update(1)
    
    return stats


def load_all_papers(data_dir: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    papers = []
    date_filter = None
    if start_date and end_date:
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            date_filter = (start_dt, end_dt)
            print(f"Date filter: {start_dt} ~ {end_dt}")
        except Exception as e:
            print(f"[Warning] 日期格式错误: {e}")
    
    # 获取所有匹配的文件
    files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    print(f"找到 {len(files)} 个数据文件")
    
    # 使用进度条显示文件加载进度
    with tqdm(total=len(files), desc="加载数据文件", unit="个") as pbar:
        for file in files:
            fname = os.path.basename(file)
            pbar.set_postfix_str(f"当前: {fname}")
            
            if fname.endswith('.jsonl'):
                date_part = fname[:-6]
                try:
                    file_dt = datetime.strptime(date_part, '%Y-%m-%d')
                    if date_filter:
                        if not (date_filter[0] <= file_dt <= date_filter[1]):
                            pbar.update(1)
                            continue
                except Exception as e:
                    print(f"  Date parse error: {e}")
                    pass
            
            # 计算文件行数用于进度显示
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                with open(file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            paper = json.loads(line)
                            papers.append(paper)
                        except Exception as e:
                            print(f"  Error parsing line {i} in {fname}: {e}")
            except Exception as e:
                print(f"  Error reading file {fname}: {e}")
            
            pbar.update(1)
    
    print(f"总共加载了 {len(papers)} 篇论文")
    return papers
# 1. 读取所有jsonl文件，合并为论文列表
# def load_all_papers(data_dir: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
#     papers = []
#     # 解析日期范围
#     date_filter = None
#     if start_date and end_date:
#         try:
#             start_dt = datetime.strptime(start_date, '%Y-%m-%d')
#             end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#             date_filter = (start_dt, end_dt)
#         except Exception as e:
#             print(f"[Warning] 日期格式错误: {e}")
#     for file in glob.glob(os.path.join(data_dir, '*.jsonl')):
#         # 文件名如2025-06-10.jsonl
#         fname = os.path.basename(file)
#         if fname.endswith('.jsonl'):
#             date_part = fname[:-6]
#             try:
#                 file_dt = datetime.strptime(date_part, '%Y-%m-%d')
#                 if date_filter:
#                     if not (date_filter[0] <= file_dt <= date_filter[1]):
#                         continue
#             except Exception:
#                 pass  # 非日期文件名，跳过
#         with open(file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     paper = json.loads(line)
#                     papers.append(paper)
#                 except Exception as e:
#                     print(f"Error parsing line in {file}: {e}")
#     return papers

# 常见领域/任务/方法同义词映射
SYNONYM_MAP = {
    # 一级
    '计算机视觉': ['视觉', 'CV', 'computer vision'],
    '自然语言处理': ['NLP', '语言处理', 'natural language processing'],
    '多模态学习': ['多模态', 'multimodal learning'],
    '机器学习': ['ML', 'machine learning'],
    '深度学习': ['DL', 'deep learning'],
    # 二级/三级
    '图像生成': ['生成模型', 'image generation'],
    '目标检测': ['object detection'],
    '图像分割': ['image segmentation'],
    '图像分类': ['image classification'],
    '文本生成': ['text generation'],
    '对话系统': ['chatbot', 'dialogue system'],
    '知识图谱': ['knowledge graph'],
    '情感分析': ['sentiment analysis'],
    '机器翻译': ['machine translation'],
    '问答系统': ['question answering'],
    '信息抽取': ['information extraction'],
    '图像去噪': ['denoising'],
    '图像超分辨率': ['super resolution'],
    '姿态估计': ['pose estimation'],
    '三维重建': ['3d reconstruction'],
    '自动代码补全与程序修复': ['code completion', 'program repair'],
    '多媒体取证与深度伪造检测': ['deepfake detection', 'media forensics'],
    # ...可继续扩展
}

# 反向映射：同义词->主名
SYNONYM_REVERSE = {}
for main, syns in SYNONYM_MAP.items():
    for s in syns:
        SYNONYM_REVERSE[s] = main
    SYNONYM_REVERSE[main] = main

def normalize_name(name: str) -> str:
    name = name.strip().lower()
    return SYNONYM_REVERSE.get(name, name)

# 节点类型映射
NODE_TYPE_MAP = {
    # 一级领域
    '计算机视觉': 'field',
    '自然语言处理': 'field',
    '多模态学习': 'field',
    '机器学习': 'field',
    '深度学习': 'field',
    # 任务
    '图像生成': 'task',
    '目标检测': 'task',
    '图像分割': 'task',
    '图像分类': 'task',
    '文本生成': 'task',
    '对话系统': 'task',
    '情感分析': 'task',
    '机器翻译': 'task',
    '问答系统': 'task',
    '信息抽取': 'task',
    '图像去噪': 'task',
    '图像超分辨率': 'task',
    '姿态估计': 'task',
    '三维重建': 'task',
    # 方法
    '知识图谱': 'method',
    # 应用
    '自动代码补全与程序修复': 'application',
    '多媒体取证与深度伪造检测': 'evaluation',
    # ...可继续扩展
}

def get_node_type(name: str) -> str:
    return NODE_TYPE_MAP.get(name, 'unknown')

# 2. 多级领域聚合数据结构
def build_hierarchical_stats(papers: List[Dict[str, Any]], max_level: int = 3) -> Dict:
    """
    支持多级聚合，max_level控制聚合到第几层
    """
    from collections import defaultdict
    def make_level():
        return {'__papers__': [], '__subs__': defaultdict(make_level)}
    stats = defaultdict(make_level)
    
    # 使用进度条显示层次统计构建进度
    with tqdm(total=len(papers), desc="构建层次统计", unit="篇") as pbar:
        for paper in papers:
            path = paper.get('classification_path', '')
            if not path:
                pbar.update(1)
                continue
            parts = path.split(' → ')
            if len(parts) == 0:
                pbar.update(1)
                continue
            key = tuple(parts[:max_level])
            if len(key) == 1:
                stats[key[0]]['__papers__'].append(paper)
            elif len(key) == 2:
                stats[key[0]]['__subs__'][key[1]]['__papers__'].append(paper)
            elif len(key) >= 3:
                stats[key[0]]['__subs__'][key[1]]['__subs__'][key[2]]['__papers__'].append(paper)
            pbar.update(1)
    return stats

# 3. 同义词/近义词合并
def synonym_merge(stats: Dict) -> Dict:
    def merge_level(d, level=1):
        if level == 3:
            merged = defaultdict(list)
        elif level == 2:
            merged = defaultdict(lambda: defaultdict(list))
        else:
            merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        name_map = {}
        for k, v in d.items():
            norm_k = normalize_name(k)
            name_map.setdefault(norm_k, []).append(k)
        for norm_k, orig_keys in name_map.items():
            # 合并所有同义节点的论文
            if level == 1:
                papers = []
                for ok in orig_keys:
                    papers.extend(d[ok].get('__papers__', []))
                merged[norm_k]['__papers__'] = papers
                # 递归合并二级
                sub = defaultdict(lambda: defaultdict(list))
                for ok in orig_keys:
                    for k2, v2 in d[ok].items():
                        if k2 == '__papers__': continue
                        sub[k2] = v2
                merged[norm_k].update(merge_level(sub, level=2))
                merged[norm_k]['__merged_from__'] = orig_keys
            elif level == 2:
                for ok in orig_keys:
                    for k2, v2 in d[ok].items():
                        norm_k2 = normalize_name(k2)
                        merged[norm_k2]['__papers__'].extend(v2.get('__papers__', []))
                        # 递归合并三级
                        sub = defaultdict(list)
                        for k3, v3 in v2.items():
                            if k3 == '__papers__': continue
                            sub[k3] = v3
                        merged[norm_k2].update(merge_level(sub, level=3))
                        merged[norm_k2]['__merged_from__'] = orig_keys
            elif level == 3:
                for ok in orig_keys:
                    merged[norm_k].extend(d[ok])
        return merged
    return merge_level(stats, level=1)

# 4. 节点类型标注
def type_annotation(stats: Dict) -> Dict:
    def annotate_level(d, level=1):
        for k, v in d.items():
            if level == 1:
                v['__type__'] = get_node_type(k)
                for k2, v2 in v.items():
                    if k2.startswith('__'): continue
                    annotate_level({k2: v2}, level=2)
            elif level == 2:
                v['__type__'] = get_node_type(k)
                for k3, v3 in v.items():
                    if k3.startswith('__'): continue
                    annotate_level({k3: v3}, level=3)
            elif level == 3:
                # 三级节点类型可选，默认unknown
                pass
        return d
    return annotate_level(stats, level=1)

def merge_low_freq_nodes(stats: Dict, min_paper_count: int = 5) -> Dict:
    def merge_level(d, level=1):
        if level > 2:
            return d
        # 统计每个节点的论文数
        node_counts = {}
        for k, v in d.items():
            if k.startswith('__'):
                continue
            if level == 1:
                count = len(v.get('__papers__', []))
            elif level == 2:
                count = len(v.get('__papers__', []))
            node_counts[k] = count
        # 找出低频节点
        low_freq = [k for k, c in node_counts.items() if c < min_paper_count]
        high_freq = [k for k in d if not k.startswith('__') and k not in low_freq]
        # 合并低频节点
        if low_freq:
            other_node = {'__papers__': [], '__type__': 'other', '__merged_from__': low_freq}
            for k in low_freq:
                other_node['__papers__'].extend(d[k].get('__papers__', []))
                # 递归合并下级
                for k2, v2 in d[k].items():
                    if k2.startswith('__'): continue
                    if '__subs__' not in other_node:
                        other_node['__subs__'] = {}
                    if k2 not in other_node['__subs__']:
                        other_node['__subs__'][k2] = v2
                    else:
                        # 合并同名子节点
                        other_node['__subs__'][k2]['__papers__'].extend(v2.get('__papers__', []))
            # 保留高频节点
            merged = {k: v for k, v in d.items() if k in high_freq or k.startswith('__')}
            merged['其他'] = other_node
        else:
            merged = d
        # 递归处理高频节点
        for k in high_freq:
            for k2, v2 in merged[k].items():
                if k2.startswith('__'): continue
                merged[k][k2] = merge_level({k2: v2}, level=level+1)[k2]
        return merged
    return merge_level(stats, level=1)

# 5. 输出多级聚合统计
def save_stats(stats: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hierarchical_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# 6. 保存分类结果到jsonl文件
def save_classified_papers(papers: List[Dict[str, Any]], output_dir: str, original_files: List[str]):
    """将分类结果保存回原始的jsonl文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 按原始文件名分组论文
    papers_by_file = defaultdict(list)
    for paper in papers:
        # 尝试从论文数据中恢复原始文件名
        # 这里假设论文有日期信息，我们可以根据日期匹配原始文件
        paper_date = paper.get('date') or paper.get('published_date')
        if paper_date:
            # 提取日期部分
            try:
                if isinstance(paper_date, str):
                    # 如果是字符串格式，尝试解析
                    from datetime import datetime
                    if 'T' in paper_date:
                        dt = datetime.fromisoformat(paper_date.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(paper_date, '%Y-%m-%d')
                    file_date = dt.strftime('%Y-%m-%d')
                else:
                    file_date = paper_date.strftime('%Y-%m-%d')
                papers_by_file[f"{file_date}.jsonl"].append(paper)
            except:
                # 如果无法解析日期，使用默认文件名
                papers_by_file["classified_papers.jsonl"].append(paper)
        else:
            papers_by_file["classified_papers.jsonl"].append(paper)
    
    # 保存到对应的文件
    for filename, file_papers in papers_by_file.items():
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in file_papers:
                json.dump(paper, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(file_papers)} classified papers to {output_file}")
    
    # 同时保存一个合并的文件
    merged_file = os.path.join(output_dir, "all_classified_papers.jsonl")
    with open(merged_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            json.dump(paper, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(papers)} total classified papers to {merged_file}")

# 7. 统计分类结果
def print_classification_stats(papers: List[Dict[str, Any]]):
    """打印分类统计信息"""
    total_papers = len(papers)
    classified_papers = [p for p in papers if p.get('classification_path')]
    failed_papers = total_papers - len(classified_papers)
    
    print(f"\n=== 分类统计 ===")
    print(f"总论文数: {total_papers}")
    print(f"成功分类: {len(classified_papers)}")
    print(f"分类失败: {failed_papers}")
    if total_papers > 0:
        print(f"成功率: {len(classified_papers)/total_papers*100:.1f}%")
    else:
        print("成功率: 0.0% (无数据)")
    
    # 统计各领域分布
    domain_counts = Counter()
    for paper in classified_papers:
        path = paper.get('classification_path', '')
        if path:
            parts = path.split(' → ')
            if len(parts) >= 2:
                domain_counts[parts[1]] += 1
    
    print(f"\n=== 领域分布 ===")
    for domain, count in domain_counts.most_common(10):
        print(f"{domain}: {count} 篇")
    
    return len(classified_papers), failed_papers

# 标准领域词表
STANDARD_LEVEL1 = [
    "计算机视觉", "自然语言处理", "语音与音频处理", "机器学习", "强化学习", "多模态学习", "机器人学", "人机交互", "医学图像", "大语言模型", "数据可视化", "其它"
]
STANDARD_LEVEL2 = [
    "目标检测", "图像分割", "生成模型", "问答系统", "对话系统", "语音识别", "视频理解", "代码生成", "医学图像分割", "小样本学习", "三维重建", "其它"
]

# 领域归一化映射（可扩展）
DOMAIN_SYNONYMS = {
    "目标检测与跟踪": "目标检测",
    "目标检测和跟踪": "目标检测",
    "视觉-语言模型": "多模态学习",
    "人工智能与计算机视觉": "计算机视觉",
    "人工智能与图像处理": "计算机视觉",
    "人工智能与自然语言处理": "自然语言处理",
    "人工智能": "其它",  # 二级领域禁止用人工智能
    "AI": "其它",
    # 新增映射
    "三维人类动画": "计算机视觉",
    "三维重建": "计算机视觉", 
    "4D SLAM": "计算机视觉",
    "动态场景重建": "计算机视觉",
    "高维数据可视化": "数据可视化",
    "可视分析": "数据可视化",
    "虚拟现实": "人机交互",
    "立体视觉": "人机交互",
    "医学图像分割": "医学图像分割",
    "磁粒子成像": "医学图像",
    "蛋白质序列": "机器学习",
    "扩散模型": "生成模型",
    "图像生成": "生成模型",
    "视频生成": "生成模型",
    "多模态风格迁移": "生成模型",
    "多token预测": "大语言模型",
    "课程学习": "机器学习",
    "多语言": "自然语言处理",
    "长文本建模": "自然语言处理",
    "位置编码": "自然语言处理",
    "句法结构": "自然语言处理",
    "叙事分类": "自然语言处理",
    "关系抽取": "自然语言处理",
    "自然语言解释": "自然语言处理",
    "自然语言推理": "自然语言处理",
    "开放域对话": "对话系统",
    "多轮对话": "对话系统",
    "检索增强生成": "问答系统",
    "RAG": "问答系统",
    "多语言推理": "问答系统",
    "开放世界": "计算机视觉",
    "自我中心": "计算机视觉",
    "活动识别": "计算机视觉",
    "3D高斯": "计算机视觉",
    "渲染压缩": "计算机视觉",
    "CAD重建": "多模态学习",
    "多模态大模型": "多模态学习",
    "被动辅助": "多模态学习",
    "多模态交互": "多模态学习",
    "多模态风格": "多模态学习",
    "视频文本理解": "视频理解",
    "视频表征学习": "视频理解",
    "开放世界活动": "视频理解",
    "句子重音": "语音与音频处理",
    "重音检测": "语音与音频处理",
    "重音推理": "语音与音频处理",
    "自动作文评分": "自然语言处理",
    "作文评分": "自然语言处理",
    "系统综述": "大语言模型",
    "医学领域": "医学图像",
    "对象计数": "目标检测",
    "指代表达": "目标检测",
    "计数": "目标检测",
    "非刚性": "计算机视觉",
    "动态场景": "计算机视觉",
    "SLAM": "计算机视觉",
    "质量评估": "计算机视觉",
    "动画质量": "计算机视觉",
    "距离感知": "人机交互",
    "立体视觉误差": "人机交互",
    "数据缺失": "数据可视化",
    "高维数据": "数据可视化",
    "缺失可视": "数据可视化",
    "多目标强化学习": "强化学习",
    "多目标": "强化学习",
    "程序分析": "代码生成",
    "反馈": "代码生成",
    "企业气候": "问答系统",
    "气候披露": "问答系统",
    "图像矢量化": "生成模型",
    "分层编辑": "生成模型",
    "矢量化": "生成模型",
    "预训练": "大语言模型",
    "token预测": "大语言模型",
    "课程学习策略": "机器学习",
    "重音检测与推理": "语音与音频处理",
    "重音检测": "语音与音频处理",
    "重音推理": "语音与音频处理",
    "句法结构变异": "自然语言处理",
    "变异分析": "自然语言处理",
    "多语言开放域": "对话系统",
    "开放域": "对话系统",
    "多语言叙事": "自然语言处理",
    "叙事分类": "自然语言处理",
    "多语言推理能力": "问答系统",
    "推理能力": "问答系统",
    "多模态CAD": "多模态学习",
    "CAD": "多模态学习",
    "多模态大模型在对话": "多模态学习",
    "对话场景": "多模态学习",
    "被动辅助应用": "多模态学习",
    "辅助应用": "多模态学习",
    "多模态风格迁移": "生成模型",
    "风格迁移": "生成模型",
    "医学图像分割": "医学图像分割",
    "医学图像": "医学图像分割",
    "重建算法": "医学图像",
    "磁粒子": "医学图像",
    "成像重建": "医学图像",
    "蛋白质序列生成": "机器学习",
    "序列生成": "机器学习",
    "自然语言推理解释": "自然语言处理",
    "推理解释": "自然语言处理",
    "解释分类": "自然语言处理",
    "3D高斯渲染": "计算机视觉",
    "高斯渲染": "计算机视觉",
    "渲染压缩": "计算机视觉",
    "压缩": "计算机视觉",
    "小样本学习": "小样本学习",
    "小样本": "小样本学习",
    "多语言推理能力评估": "问答系统",
    "推理能力评估": "问答系统",
    "能力评估": "问答系统",
    "多模态CAD重建": "多模态学习",
    "CAD重建": "多模态学习",
    "多模态问答": "多模态学习",
    "问答": "问答系统",
    # ...可继续扩展
}

# 领域路径归一化与交叉领域处理
import re
def normalize_classification_path(path: str) -> str:
    # 拆分“与”“和”“及”等连接词，优先保留标准词表中的最细分领域
    parts = re.split(r'[→\->]|[、,，]', path)
    parts = [p.strip() for p in parts if p.strip()]
    norm_parts = []
    # 一级领域强制为“计算机科学”
    if parts and parts[0] != "计算机科学":
        norm_parts.append("计算机科学")
    else:
        norm_parts.append(parts[0] if parts else "计算机科学")
    # 二级领域
    level1 = None
    for p in parts[1:]:
        # 替换同义词
        p_norm = DOMAIN_SYNONYMS.get(p, p)
        if p_norm in STANDARD_LEVEL1:
            level1 = p_norm
            break
    if not level1:
        # 兜底：如有交叉领域，优先保留标准词表中最细分的
        for p in parts[1:]:
            for std in STANDARD_LEVEL1:
                if std in p:
                    level1 = std
                    break
            if level1:
                break
    if not level1:
        level1 = "其它"
    norm_parts.append(level1)
    # 三级领域
    level2 = None
    for p in parts[2:]:
        p_norm = DOMAIN_SYNONYMS.get(p, p)
        if p_norm in STANDARD_LEVEL2:
            level2 = p_norm
            break
    if not level2:
        for p in parts[2:]:
            for std in STANDARD_LEVEL2:
                if std in p:
                    level2 = std
                    break
            if level2:
                break
    if not level2:
        level2 = None
    if level2:
        norm_parts.append(level2)
    # 四级领域（可选，保留原始）
    if len(parts) > 3:
        norm_parts.extend(parts[3:])
    return " → ".join([p for p in norm_parts if p])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='领域多级聚合分析')
    parser.add_argument('--start-date', type=str, help='起始日期 YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='结束日期 YYYY-MM-DD')
    args = parser.parse_args()

    papers = load_all_papers(DATA_DIR, start_date=args.start_date, end_date=args.end_date)
    print(f"Loaded {len(papers)} papers.")
    
    # 检查数据完整性
    print("\n=== 数据完整性检查 ===")
    integrity_stats = check_paper_data_integrity(papers)
    print(f"总论文数: {integrity_stats['total']}")
    print(f"有效论文数: {integrity_stats['valid']}")
    print(f"缺少标题: {integrity_stats['missing_title']}")
    print(f"缺少摘要: {integrity_stats['missing_abstract']}")
    print(f"标题摘要都缺少: {integrity_stats['missing_both']}")
    print(f"空标题: {integrity_stats['empty_title']}")
    print(f"空摘要: {integrity_stats['empty_abstract']}")
    print("=" * 30)

    # 实例化新版OpenAI client
    client = None
    if OpenAI and OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        print("[Error] OpenAI API Key或openai包未配置，无法进行分类。")

    # 批量补全classification_path
    cache = load_classification_cache()
    cache_lock = threading.Lock()
    
    # 统计需要分类的论文数量
    papers_to_classify = [p for p in papers if not p.get('classification_path') or not all(k in p for k in ['keywords', 'methods', 'problems'])]
    already_classified = len(papers) - len(papers_to_classify)
    
    print(f"\n=== 分类进度 ===")
    print(f"总论文数: {len(papers)}")
    print(f"已分类论文: {already_classified}")
    print(f"待分类论文: {len(papers_to_classify)}")
    print("=" * 30)
    
    # 多线程并发分类
    def classify_and_update(paper):
        # 线程安全地操作cache
        with cache_lock:
            paper_id = paper.get('id') or paper.get('title')
            if paper_id in cache and cache[paper_id].get('classification_path') and all(
                k in cache[paper_id] for k in ['keywords', 'methods', 'problems']
            ):
                paper['keywords'] = cache[paper_id]['keywords']
                paper['methods'] = cache[paper_id]['methods']
                paper['problems'] = cache[paper_id]['problems']
                paper['classification_path'] = normalize_classification_path(cache[paper_id]['classification_path'])
                return paper['classification_path']
        # 调用API
        result = classify_paper_with_openai(paper, cache, client)
        # 线程安全地写入cache
        with cache_lock:
            paper_id = paper.get('id') or paper.get('title')
            if paper_id and paper.get('classification_path'):
                cache[paper_id] = {
                    'classification_path': paper['classification_path'],
                    'keywords': paper.get('keywords', []),
                    'methods': paper.get('methods', []),
                    'problems': paper.get('problems', [])
                }
        return result

    max_workers = min(8, len(papers_to_classify)) if len(papers_to_classify) > 0 else 1
    with tqdm(total=len(papers_to_classify), desc="论文分类进度", unit="篇") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(classify_and_update, paper): paper for paper in papers_to_classify}
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print(f"[Thread Error] 论文 {paper.get('id') or paper.get('title', 'Unknown')[:30]} 分类异常: {exc}")
                pbar.update(1)
                paper_id = paper.get('id') or paper.get('title', 'Unknown')[:30]
                pbar.set_postfix_str(f"当前: {paper_id}")

    save_classification_cache(cache) # 保存缓存

    # 统计分类结果
    classified_count, failed_count = print_classification_stats(papers)
    
    # 保存分类结果到jsonl文件
    original_files = glob.glob(os.path.join(DATA_DIR, '*.jsonl'))
    save_classified_papers(papers, OUTPUT_DIR, original_files)
    
    # 构建层次统计
    stats = build_hierarchical_stats(papers, max_level=3)
    stats = synonym_merge(stats)
    stats = type_annotation(stats)
    # 暂时跳过低频节点合并，避免过度合并
    # stats = merge_low_freq_nodes(stats, min_paper_count=5)
    save_stats(stats, OUTPUT_DIR)
    print(f"Hierarchical stats saved to {OUTPUT_DIR}/hierarchical_stats.json")
    
    print(f"\n=== 处理完成 ===")
    print(f"成功分类: {classified_count} 篇论文")
    print(f"分类失败: {failed_count} 篇论文")
    print(f"分类结果已保存到: {OUTPUT_DIR}/") 