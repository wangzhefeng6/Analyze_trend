#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API论文趋势分析可视化界面
基于Streamlit和Plotly的论文趋势可视化dashboard
使用OpenAI API大模型进行论文分析
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

from gpt4o_trend_analyzer import GPT4oTrendAnalyzer

# 设置页面配置
st.set_page_config(
    page_title="论文趋势分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class TrendDashboard:
    """趋势分析可视化界面"""
    
    def __init__(self):
        # 加载.env文件
        if os.path.exists('.env'):
            load_dotenv('.env')
        
        self.analyzer = None
        
        # 使用session state保存数据，避免页面刷新时丢失
        if 'analyzed_papers' not in st.session_state:
            st.session_state.analyzed_papers = []
        if 'trends' not in st.session_state:
            st.session_state.trends = {}
        
        # 自动加载已有结果（如果存在且session state为空）
        if not st.session_state.analyzed_papers:
            self.load_existing_results()
        
        # 为了兼容性，保留原有的属性引用
        self.analyzed_papers = st.session_state.analyzed_papers
        self.trends = st.session_state.trends
        
    def initialize_analyzer(self, api_key, data_dir="data", proxy=None):
        """初始化分析器"""
        try:
            # 使用OpenAI API分析器
            self.analyzer = GPT4oTrendAnalyzer(api_key=api_key, proxy_url=proxy, data_dir=data_dir)
            return True
        except Exception as e:
            st.error(f"初始化分析器失败: {e}")
            return False
    
    def load_existing_results(self, results_dir="analysis_results"):
        """加载已有的分析结果"""
        try:
            papers_file = os.path.join(results_dir, 'analyzed_papers.json')
            trends_file = os.path.join(results_dir, 'trends_analysis.json')
            
            if os.path.exists(papers_file) and os.path.exists(trends_file):
                with open(papers_file, 'r', encoding='utf-8') as f:
                    st.session_state.analyzed_papers = json.load(f)
                
                with open(trends_file, 'r', encoding='utf-8') as f:
                    st.session_state.trends = json.load(f)
                
                # 更新实例引用
                self.analyzed_papers = st.session_state.analyzed_papers
                self.trends = st.session_state.trends
                
                return True
            return False
        except Exception as e:
            st.error(f"加载结果失败: {e}")
            return False
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("📊论文趋势分析")
        
        # 从环境变量获取OpenAI API密钥
        api_key = os.getenv('OPENAI_API_KEY', '')
        proxy_url = os.getenv('PROXY_URL', '')  # 保留用于兼容性，OpenAI客户端会忽略
        # 自动检测数据目录（不显示状态）
        data_dir = "../data"
        if not os.path.exists(data_dir):
            # 尝试其他可能的路径
            possible_paths = ["data", "../../data"]
            for path in possible_paths:
                if os.path.exists(path):
                    data_dir = path
                    break
        
        # 类别选择
        st.sidebar.subheader("📚 论文类别选择")
        
        # 使用默认的常见类别（基于arXiv热门分类）
        available_categories = {
            'cs.CV': 150,    # 计算机视觉
            'cs.AI': 120,    # 人工智能
            'cs.LG': 100,    # 机器学习
            'cs.CL': 90,     # 计算语言学
            'cs.RO': 50,     # 机器人学
            'cs.GR': 30,     # 计算机图形学
            'cs.IR': 25,     # 信息检索
            'cs.HC': 20,     # 人机交互
            'eess.IV': 40,   # 图像与视频处理
            'eess.SP': 15,   # 信号处理
        }
        
        # 类别映射（中文显示）
        category_mapping = {
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
        
        # 创建类别选择选项
        category_options = []
        for cat, count in sorted(available_categories.items(), key=lambda x: x[1], reverse=True)[:15]:
            display_name = category_mapping.get(cat, cat)
            category_options.append(f"{display_name} ({cat}) - {count}篇")
        
        # 多选框
        selected_categories_display = st.sidebar.multiselect(
            "选择要分析的论文类别",
            options=category_options,
            default=category_options[:3] if category_options else [],
            help="可选择多个类别进行对比分析"
        )
        
        # 提取实际的类别代码
        selected_categories = []
        for display_name in selected_categories_display:
            # 从显示名称中提取类别代码
            import re
            match = re.search(r'\(([^)]+)\)', display_name)
            if match:
                selected_categories.append(match.group(1))
        
        # 显示选中的类别
        if selected_categories:
            st.sidebar.info(f"已选择 {len(selected_categories)} 个类别")
        else:
            st.sidebar.warning("⚠️ 请至少选择一个类别")
        
        # 日期范围选择
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30),
                help="分析的起始日期"
            )
        with col2:
            end_date = st.date_input(
                "结束日期", 
                value=datetime.now(),
                help="分析的结束日期"
            )
        
        # 高级设置
        with st.sidebar.expander("⚙️ 高级设置"):
            max_workers = st.slider(
                "并发数",
                min_value=5,
                max_value=20,
                value=15,
                help="控制OpenAI API的并发请求数，建议10-20"
            )

        
        # 开始智能分析按钮
        if st.sidebar.button("🧠 开始智能分析"):
            if not api_key:
                st.sidebar.error("❌ 请在.env文件中设置OPENAI_API_KEY")
                return
            
            if not os.path.exists(data_dir):
                st.sidebar.error("❌ 请先确保数据目录存在")
                return
            
            if not selected_categories:
                st.sidebar.error("❌ 请至少选择一个论文类别")
                return
            
            if not self.initialize_analyzer(api_key, data_dir, proxy_url):
                return
            
            # 运行完整的智能分析流程
            with st.spinner(f"🧠 正在进行大模型智能分析 {len(selected_categories)} 个类别的论文..."):
                try:
                    # 第一步：基础论文分析
                    st.info("📊 步骤1/2: 正在使用大模型分析论文...")
                    analyzed_papers, trends = self.analyzer.run_full_analysis(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        max_workers=max_workers,
                        categories=selected_categories,
                        include_category_analysis=True
                    )
                    
                    # 保存基础分析结果到session state
                    st.session_state.analyzed_papers = analyzed_papers
                    st.session_state.trends = trends
                    st.session_state.selected_categories = selected_categories
                    st.session_state.start_date = start_date.strftime('%Y-%m-%d')
                    st.session_state.end_date = end_date.strftime('%Y-%m-%d')
                    
                    # 更新实例引用
                    self.analyzed_papers = st.session_state.analyzed_papers
                    self.trends = st.session_state.trends
                    
                    # 第二步：智能总结
                    st.info("🤖 步骤2/2: 正在生成智能总结...")
                    summary = self.analyzer.summarize_trends_with_gpt4o(
                        trends,
                        categories=selected_categories,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    # 保存智能总结到session state
                    st.session_state.intelligent_summary = summary
                    
                    st.sidebar.success(f"✅ 智能分析完成! 共分析 {len(self.analyzed_papers)} 篇论文，并生成了AI智能总结")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ 智能分析失败: {e}")
                    st.sidebar.write("详细错误信息:")
                    st.sidebar.code(str(e))
        
        # 智能分析状态显示（仅在有分析结果时显示）
        if st.session_state.analyzed_papers and st.session_state.trends:
            st.sidebar.markdown("---")
            st.sidebar.subheader("✅ 智能分析状态")
            st.sidebar.success(f"📊 已分析 {len(st.session_state.analyzed_papers)} 篇论文")
            
            if st.session_state.get('intelligent_summary'):
                st.sidebar.success("🤖 AI智能总结已生成")
            else:
                st.sidebar.info("💡 智能总结未生成，请重新运行智能分析")
    

    
    def render_keyword_analysis(self):
        """渲染关键词分析页面"""
        st.markdown('<div class="main-header">🔍 关键词趋势分析</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analyzed_papers:
            st.info("👆 请先在侧边栏中加载数据或开始智能分析")
            return
        
        # 获取分析数据
        analyzed_papers = st.session_state.analyzed_papers
        
        # 关键词统计
        all_keywords = []
        keyword_by_date = defaultdict(Counter)
        
        for paper in analyzed_papers:
            date = paper.get('date', '')
            keywords = paper.get('keywords', [])
            all_keywords.extend(keywords)
            
            for keyword in keywords:
                if keyword:
                    keyword_by_date[date][keyword] += 1
        
        # 最热门关键词
        st.markdown('<div class="sub-header">🔥 最热门关键词</div>', 
                   unsafe_allow_html=True)
        
        keyword_counts = Counter(all_keywords)
        # 动态确定显示数量：最少5个，最多30个
        total_keywords = len(keyword_counts)
        display_count = min(max(total_keywords, 5), 30)
        top_keywords = keyword_counts.most_common(display_count)
        
        if top_keywords:
            df_keywords = pd.DataFrame(top_keywords, columns=['关键词', '出现次数'])
            
            fig_bar = px.bar(
                df_keywords,
                x='出现次数',
                y='关键词',
                orientation='h',
                title=f"热门关键词 (共{total_keywords}个，显示前{len(top_keywords)}个)",
                color='出现次数',
                color_continuous_scale='Viridis'
            )
            
            # 根据显示的关键词数量动态调整图表高度
            chart_height = max(400, len(df_keywords) * 20)
            fig_bar.update_layout(height=chart_height, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 关键词云
            st.markdown('<div class="sub-header">☁️ 关键词云</div>', 
                       unsafe_allow_html=True)
            
            # 使用简单的表格显示关键词权重
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("高频关键词")
                # 动态显示数量，最多15个
                high_freq_count = min(len(df_keywords), 15)
                st.dataframe(df_keywords.head(high_freq_count), use_container_width=True)
            
            with col2:
                st.subheader("新兴关键词")
                # 计算最近几天的新兴关键词
                total_dates = len(keyword_by_date)
                recent_days = min(max(total_dates // 3, 3), 7)  # 动态确定最近天数：总天数的1/3，但至少3天，最多7天
                recent_dates = sorted(keyword_by_date.keys())[-recent_days:]
                recent_keywords = Counter()
                for date in recent_dates:
                    recent_keywords.update(keyword_by_date[date])
                
                # 排除已有的热门关键词
                emerging_keywords = []
                high_freq_threshold = min(len(top_keywords), 15)
                top_keyword_set = set([k for k, _ in top_keywords[:high_freq_threshold]])
                
                # 动态确定新兴关键词的阈值
                min_count = max(1, total_dates // 10)  # 至少出现总天数的1/10次
                max_emerging = min(max(len(recent_keywords) // 4, 5), 15)  # 显示最多15个新兴关键词
                
                for keyword, count in recent_keywords.most_common(max_emerging * 2):
                    if keyword not in top_keyword_set and count >= min_count:
                        emerging_keywords.append((keyword, count))
                
                if emerging_keywords:
                    display_emerging = emerging_keywords[:max_emerging]
                    df_emerging = pd.DataFrame(display_emerging, 
                                             columns=['关键词', '出现次数'])
                    st.dataframe(df_emerging, use_container_width=True)
                    st.caption(f"基于最近{recent_days}天数据，共发现{len(emerging_keywords)}个新兴关键词")
                else:
                    st.info("暂无新兴关键词")
        
        # 关键词时间趋势
        st.markdown('<div class="sub-header">📊 关键词时间趋势</div>', 
                   unsafe_allow_html=True)
        
        # 选择要分析的关键词 - 动态数量
        available_options = min(len(top_keywords), 25)  # 最多提供25个选项
        default_count = min(len(top_keywords), 5)  # 默认选择前5个或全部（如果少于5个）
        
        selected_keywords = st.multiselect(
            "选择要分析的关键词",
            options=[k for k, _ in top_keywords[:available_options]],
            default=[k for k, _ in top_keywords[:default_count]],
            help=f"选择要查看时间趋势的关键词 (可选择前{available_options}个热门关键词)"
        )
        
        if selected_keywords:
            # 准备时间序列数据
            dates = sorted(keyword_by_date.keys())
            
            fig_trend = go.Figure()
            
            for keyword in selected_keywords:
                counts = [keyword_by_date[date][keyword] for date in dates]
                fig_trend.add_trace(
                    go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name=keyword,
                        line=dict(width=2),
                        marker=dict(size=4)
                    )
                )
            
            fig_trend.update_layout(
                title="关键词时间趋势",
                xaxis_title="日期",
                yaxis_title="出现次数",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def render_method_analysis(self):
        """渲染方法分析页面"""
        st.markdown('<div class="main-header">⚙️ 技术方法分析</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analyzed_papers:
            st.info("👆 请先在侧边栏中加载数据或开始智能分析")
            return
        
        # 获取分析数据
        analyzed_papers = st.session_state.analyzed_papers
        
        # 方法统计
        all_methods = []
        method_by_date = defaultdict(Counter)
        method_by_category = defaultdict(Counter)
        
        for paper in analyzed_papers:
            date = paper.get('date', '')
            methods = paper.get('methods', [])
            categories = paper.get('categories', [])
            
            all_methods.extend(methods)
            
            main_category = categories[0] if categories else 'unknown'
            category_name = self.analyzer.category_mapping.get(main_category, main_category) if self.analyzer else main_category
            
            for method in methods:
                if method:
                    method_by_date[date][method] += 1
                    method_by_category[category_name][method] += 1
        
        # 热门技术方法
        st.markdown('<div class="sub-header">🛠️ 热门技术方法</div>', 
                   unsafe_allow_html=True)
        
        method_counts = Counter(all_methods)
        top_methods = method_counts.most_common(15)
        
        if top_methods:
            df_methods = pd.DataFrame(top_methods, columns=['技术方法', '使用次数'])
            
            fig_methods = px.bar(
                df_methods,
                x='使用次数',
                y='技术方法',
                orientation='h',
                title="Top 15 热门技术方法",
                color='使用次数',
                color_continuous_scale='Blues'
            )
            
            fig_methods.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_methods, use_container_width=True)
        

        
        # 方法时间趋势
        st.markdown('<div class="sub-header">📈 方法发展趋势</div>', 
                   unsafe_allow_html=True)
        
        selected_methods = st.multiselect(
            "选择要分析的技术方法",
            options=[m for m, _ in top_methods[:15]],
            default=[m for m, _ in top_methods[:4]],
            help="选择要查看发展趋势的技术方法"
        )
        
        if selected_methods:
            dates = sorted(method_by_date.keys())
            
            fig_method_trend = go.Figure()
            
            for method in selected_methods:
                counts = [method_by_date[date][method] for date in dates]
                fig_method_trend.add_trace(
                    go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name=method,
                        line=dict(width=2),
                        marker=dict(size=4)
                    )
                )
            
            fig_method_trend.update_layout(
                title="技术方法发展趋势",
                xaxis_title="日期",
                yaxis_title="使用次数",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_method_trend, use_container_width=True)
    
    def render_problem_analysis(self):
        """渲染研究问题分析页面"""
        st.markdown('<div class="main-header">❓ 研究问题分析</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analyzed_papers:
            st.info("👆 请先在侧边栏中加载数据或开始智能分析")
            return
        
        # 获取分析数据
        analyzed_papers = st.session_state.analyzed_papers
        
        # 问题统计
        all_problems = []
        problem_by_date = defaultdict(Counter)
        problem_by_category = defaultdict(Counter)
        
        for paper in analyzed_papers:
            date = paper.get('date', '')
            problems = paper.get('problems', [])
            categories = paper.get('categories', [])
            
            all_problems.extend(problems)
            
            main_category = categories[0] if categories else 'unknown'
            category_name = self.analyzer.category_mapping.get(main_category, main_category) if self.analyzer else main_category
            
            for problem in problems:
                if problem:
                    problem_by_date[date][problem] += 1
                    problem_by_category[category_name][problem] += 1
        
        # 热门研究问题
        st.markdown('<div class="sub-header">🔥 热门研究问题</div>', 
                   unsafe_allow_html=True)
        
        problem_counts = Counter(all_problems)
        top_problems = problem_counts.most_common(15)
        
        if top_problems:
            df_problems = pd.DataFrame(top_problems, columns=['研究问题', '关注度'])
            
            fig_problems = px.bar(
                df_problems,
                x='关注度',
                y='研究问题',
                orientation='h',
                title="Top 15 热门研究问题",
                color='关注度',
                color_continuous_scale='Reds'
            )
            
            fig_problems.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_problems, use_container_width=True)
        
        # 问题与创新性的关系
        st.markdown('<div class="sub-header">💡 问题与创新性关系</div>', 
                   unsafe_allow_html=True)
        
        # 计算每个问题的平均创新分数
        problem_scores = defaultdict(list)
        for paper in self.analyzed_papers:
            problems = paper.get('problems', [])
            score = paper.get('score', 0)
            for problem in problems:
                if problem:
                    problem_scores[problem].append(score)
        
        # 只保留出现次数>=3的问题
        problem_innovation = []
        for problem, scores in problem_scores.items():
            if len(scores) >= 3:
                avg_score = np.mean(scores)
                count = len(scores)
                problem_innovation.append((problem, avg_score, count))
        
        if problem_innovation:
            problem_innovation.sort(key=lambda x: x[1], reverse=True)
            
            df_innovation = pd.DataFrame(
                problem_innovation[:12], 
                columns=['研究问题', '平均创新分数', '论文数量']
            )
            
            fig_innovation = px.scatter(
                df_innovation,
                x='平均创新分数',
                y='研究问题',
                size='论文数量',
                title="研究问题的创新性分析",
                color='平均创新分数',
                color_continuous_scale='Viridis'
            )
            
            fig_innovation.update_layout(height=500)
            st.plotly_chart(fig_innovation, use_container_width=True)
        
        # 问题演进趋势
        st.markdown('<div class="sub-header">🔄 问题演进趋势</div>', 
                   unsafe_allow_html=True)
        
        selected_problems = st.multiselect(
            "选择要分析的研究问题",
            options=[p for p, _ in top_problems[:15]],
            default=[p for p, _ in top_problems[:4]],
            help="选择要查看演进趋势的研究问题"
        )
        
        if selected_problems:
            dates = sorted(problem_by_date.keys())
            
            fig_problem_trend = go.Figure()
            
            for problem in selected_problems:
                counts = [problem_by_date[date][problem] for date in dates]
                fig_problem_trend.add_trace(
                    go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name=problem,
                        line=dict(width=2),
                        marker=dict(size=4)
                    )
                )
            
            fig_problem_trend.update_layout(
                title="研究问题演进趋势",
                xaxis_title="日期",
                yaxis_title="关注度",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_problem_trend, use_container_width=True)
    
    def render_detailed_data(self):
        """渲染详细数据页面"""
        st.markdown('<div class="main-header">📋 详细数据</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analyzed_papers:
            st.info("👆 请先在侧边栏中加载数据或开始智能分析")
            return
        
        # 获取分析数据
        analyzed_papers = st.session_state.analyzed_papers
        
        # 数据筛选
        st.markdown('<div class="sub-header">🔍 数据筛选</div>', 
                   unsafe_allow_html=True)
        
        # 准备筛选选项
        all_categories = set()
        all_keywords = set()
        all_methods = set()
        
        for paper in analyzed_papers:
            all_categories.update(paper.get('categories', []))
            all_keywords.update(paper.get('keywords', []))
            all_methods.update(paper.get('methods', []))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_categories = st.multiselect(
                "筛选学科类别",
                options=sorted(list(all_categories)),
                help="选择要显示的学科类别"
            )
        
        with col2:
            selected_keywords = st.multiselect(
                "筛选关键词",
                options=sorted(list(all_keywords))[:50],  # 限制数量
                help="选择要筛选的关键词"
            )
        
        with col3:
            score_range = st.slider(
                "创新分数范围",
                min_value=0.0,
                max_value=5.0,
                value=(0.0, 5.0),
                step=0.1,
                help="选择创新分数范围"
            )
        
        # 应用筛选
        filtered_papers = []
        for paper in analyzed_papers:
            # 类别筛选
            if selected_categories:
                if not any(cat in paper.get('categories', []) for cat in selected_categories):
                    continue
            
            # 关键词筛选
            if selected_keywords:
                if not any(kw in paper.get('keywords', []) for kw in selected_keywords):
                    continue
            
            # 分数筛选
            score = paper.get('score', 0)
            if not (score_range[0] <= score <= score_range[1]):
                continue
            
            filtered_papers.append(paper)
        
        st.info(f"筛选后共有 {len(filtered_papers)} 篇论文")
        
        # 显示论文列表
        if filtered_papers:
            st.markdown('<div class="sub-header">📚 论文列表</div>', 
                       unsafe_allow_html=True)
            
            # 准备表格数据
            table_data = []
            for paper in filtered_papers:
                # 处理空白数据，显示友好的占位符
                keywords = paper.get('keywords', [])
                methods = paper.get('methods', [])
                problems = paper.get('problems', [])
                
                keywords_display = ', '.join(keywords[:3]) if keywords else '待分析'
                methods_display = ', '.join(methods[:2]) if methods else '待分析'
                problems_display = ', '.join(problems[:2]) if problems else '待分析'
                
                table_data.append({
                    '日期': paper.get('date', ''),
                    '标题': paper.get('title', '')[:80] + '...' if len(paper.get('title', '')) > 80 else paper.get('title', ''),
                    '主要类别': paper.get('categories', [''])[0],
                    '创新分数': paper.get('score', 0),
                    '关键词': keywords_display,
                    '主要方法': methods_display,
                    '研究问题': problems_display
                })
            
            df_table = pd.DataFrame(table_data)
            
            # 显示表格
            st.dataframe(
                df_table,
                use_container_width=True,
                height=600,
                column_config={
                    '创新分数': st.column_config.NumberColumn(
                        '创新分数',
                        min_value=0,
                        max_value=5,
                        format="%.2f"
                    )
                }
            )
            
            # 导出功能
            if st.button("📥 导出筛选结果"):
                csv = df_table.to_csv(index=False)
                st.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name=f"filtered_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """运行可视化界面"""
        # 渲染侧边栏
        self.render_sidebar()
        
        # 主要内容区域
        if not st.session_state.analyzed_papers:
            st.title("📊论文趋势分析系统")
            st.markdown("""
            ### 欢迎使用论文趋势分析系统！
            
            这个系统可以帮助您：
            - 📊 分析arXiv论文的发展趋势
            - 🔍 提取技术关键词和研究问题
            - 📈 可视化各种统计指标
            - 💡 发现研究热点和新兴方向
            
            **使用步骤：**
            1. 选择分析的时间范围
            2. 点击"🧠 开始智能分析"按钮启动完整的AI分析流程
            3. 等待分析完成后查看结果
            
            **注意事项：**
            - 分析过程需要一定时间，请耐心等待
            """)
            
            # 删除示例图表部分
            # 原有示例图表、示例数据和st.plotly_chart(fig_sample, ...)已移除

        else:
            # 创建标签页
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 关键词", "⚙️ 技术方法", "❓ 研究问题", "📋 详细数据"
            ])
            
            with tab1:
                self.render_keyword_analysis()
            
            with tab2:
                self.render_method_analysis()
            
            with tab3:
                self.render_problem_analysis()
            
            with tab4:
                self.render_detailed_data()

def main():
    """主函数"""
    dashboard = TrendDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()