import nltk
import ssl
import os

# 修复SSL证书问题（Streamlit Cloud需要）
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 自动下载NLTK资源
print("正在下载NLTK资源...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print("NLTK资源下载完成！")

# 导入其他库
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter

st.set_page_config(
    page_title="语义表示与对比分析系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #1a1a2e;
    }
    .stApp {
        background-color: #1a1a2e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #16213e;
        border-radius: 8px 8px 0px 0px;
        color: #e6e6e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: #ff6b35;
    }
    h1 {
        color: #ff6b35 !important;
    }
    h2 {
        color: #4da6ff !important;
    }
    h3 {
        color: #e6e6e6 !important;
    }
    .stButton>button {
        background-color: #1e3c72;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .card {
        background-color: #16213e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_TEXT = """Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics. It is primarily concerned with giving computers the ability to support and manipulate human language. It involves processing natural language datasets, such as text corpora or speech corpora, using either rule-based or probabilistic (i.e. statistical and, most recently, neural network-based) machine learning approaches. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled Computing Machinery and Intelligence which proposed what is now called the Turing test as a criterion of intelligence. The proposed test includes a task that involves the automated interpretation and generation of natural language. The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it is confronted with."""

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

processed_tokens = preprocess_text(DEFAULT_TEXT)
processed_text = ' '.join(processed_tokens)

with st.sidebar:
    st.markdown("## 🎯 系统帮助")
    with st.expander("📖 使用指南", expanded=True):
        st.markdown("""
        1. **传统统计模型**: 探索TF-IDF和LSA
        2. **词向量概念**: 了解词向量的基础
        """)
    st.markdown("## 📚 知识点导览")
    with st.expander("📊 核心概念", expanded=True):
        st.markdown("""
        - **词向量**: 将词映射到连续向量空间
        - **语义相似度**: 向量夹角表示词义相似性
        - **降维技术**: SVD/PCA用于可视化
        """)

tab1, tab2 = st.tabs(["📊 传统统计模型", "📚 词向量概念"])

with tab1:
    st.title("📊 传统统计模型 (TF-IDF与LSA)")
    
    with st.expander("📚 核心知识点：传统统计语义表示", expanded=True):
        st.markdown("""
        **🔹 TF-IDF (Term Frequency-Inverse Document Frequency)**
        - 词频(TF): TF(t,d) = COUNT(t,d)/∑COUNT(t',d)
        - 逆文档频率(IDF): IDF(t) = log(|D|/∑1_{t∈d})
        - TF-IDF = TF × IDF
        - 优势: 简单高效，适合信息检索
        - 局限: 忽略词序和语义关系
        
        **🔹 LSA (Latent Semantic Analysis)**
        - 基于奇异值分解(SVD)的降维技术
        - 将高维稀疏矩阵转换为低维稠密表示
        - 公式: A = UΣV^T
        - 能捕捉词义相关性，解决同义词问题
        """)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 文本输入")
        user_text = st.text_area("输入或修改文本", DEFAULT_TEXT, height=200)
        if st.button("🔄 预处理文本"):
            processed_tokens = preprocess_text(user_text)
            processed_text = ' '.join(processed_tokens)
            st.success("✅ 文本已更新！")
    
    with col2:
        st.subheader("👀 预处理预览")
        st.markdown(f'<div class="card">{" ".join(processed_tokens[:50])}...</div>', unsafe_allow_html=True)
        st.write(f"总词数: {len(processed_tokens)} | 唯一词数: {len(set(processed_tokens))}")
    
    st.markdown("---")
    st.subheader("🔢 TF-IDF 计算")
    
    try:
        documents = [processed_text]
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        tfidf_df = pd.DataFrame({
            'Word': feature_names,
            'TF-IDF Score': tfidf_scores
        }).sort_values('TF-IDF Score', ascending=False)
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("TF-IDF矩阵前10行:")
            st.dataframe(tfidf_df.head(10), hide_index=True)
        
        with col_b:
            st.write("Top 5关键词:")
            fig = px.bar(
                tfidf_df.head(5),
                x='Word',
                y='TF-IDF Score',
                color='TF-IDF Score',
                color_continuous_scale='Oranges',
                title='Top 5 TF-IDF关键词'
            )
            fig.update_layout(
                plot_bgcolor='#16213e',
                paper_bgcolor='#1a1a2e',
                font_color='#e6e6e6'
            )
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        st.subheader("📉 LSA 降维可视化")
        
        if len(feature_names) >= 2:
            try:
                word_vectors = tfidf_matrix.toarray()
                if word_vectors.shape[0] == 1:
                    word_vectors = np.vstack([word_vectors, np.zeros((1, word_vectors.shape[1]))])
                
                svd = TruncatedSVD(n_components=2, random_state=42)
                vectors_2d = svd.fit_transform(word_vectors.T)
                
                word_freq = Counter(processed_tokens)
                sizes = [word_freq.get(word, 1) * 10 for word in feature_names]
                
                df_lsa = pd.DataFrame({
                    'Word': feature_names,
                    'x': vectors_2d[:, 0],
                    'y': vectors_2d[:, 1],
                    'Size': sizes,
                    'TF-IDF': tfidf_scores
                })
                
                fig = px.scatter(
                    df_lsa,
                    x='x',
                    y='y',
                    size='Size',
                    color='TF-IDF',
                    hover_data=['Word', 'TF-IDF'],
                    text='Word',
                    title='LSA 2D词向量空间',
                    color_continuous_scale='Viridis'
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    plot_bgcolor='#16213e',
                    paper_bgcolor='#1a1a2e',
                    font_color='#e6e6e6',
                    height=600
                )
                st.plotly_chart(fig, width='stretch')
                
                st.info("🔍 观察: LSA是否将语义相关的词映射到相近位置？尝试输入包含同义词的文本观察效果。")
            except Exception as e:
                st.warning(f"可视化暂时不可用: {str(e)}")
        else:
            st.warning("词汇量不足，无法进行LSA降维")
            
    except Exception as e:
        st.error(f"错误: {str(e)}")

with tab2:
    st.title("📚 词向量基础概念演示")
    
    with st.expander("📚 核心概念", expanded=True):
        st.markdown("""
        **🔹 什么是词向量？**
        - 将单词映射到数值向量空间
        - 向量维度: 通常是50、100、200或300维
        - 语义越相似的词，向量距离越近
        
        **🔹 为什么需要词向量？**
        - 计算机不理解文本，但理解数字
        - 可以计算词与词之间的相似度
        - 支持词类比运算 (king - man + woman ≈ queen)
        
        **🔹 主流方法对比**
        - **统计方法**: TF-IDF、LSA（已实现）
        - **神经网络**: Word2Vec、GloVe、FastText（需要gensim）
        - **预训练模型**: BERT、GPT系列
        """)
    
    st.markdown("---")
    st.subheader("🔬 余弦相似度演示")
    
    word1 = st.text_input("词1", "language")
    word2 = st.text_input("词2", "processing")
    word3 = st.text_input("词3", "computer")
    
    if word1 and word2 and word3:
        np.random.seed(42)
        dim = 50
        vec1 = np.random.randn(dim)
        vec2 = np.random.randn(dim)
        vec3 = np.random.randn(dim)
        
        def cosine_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        sim12 = cosine_sim(vec1, vec2)
        sim13 = cosine_sim(vec1, vec3)
        sim23 = cosine_sim(vec2, vec3)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            color12 = "#1abc9c" if sim12 > 0.5 else "#f39c12" if sim12 > 0 else "#e74c3c"
            st.markdown(f'''
            <div class="card">
                <h4>{word1} ↔ {word2}</h4>
                <h3 style="color: {color12}">{sim12:.3f}</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            color13 = "#1abc9c" if sim13 > 0.5 else "#f39c12" if sim13 > 0 else "#e74c3c"
            st.markdown(f'''
            <div class="card">
                <h4>{word1} ↔ {word3}</h4>
                <h3 style="color: {color13}">{sim13:.3f}</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            color23 = "#1abc9c" if sim23 > 0.5 else "#f39c12" if sim23 > 0 else "#e74c3c"
            st.markdown(f'''
            <div class="card">
                <h4>{word2} ↔ {word3}</h4>
                <h3 style="color: {color23}">{sim23:.3f}</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 2D可视化")
        
        pca = PCA(n_components=2)
        all_vectors = np.vstack([vec1, vec2, vec3])
        vectors_2d = pca.fit_transform(all_vectors)
        
        df_viz = pd.DataFrame({
            'Word': [word1, word2, word3],
            'x': vectors_2d[:, 0],
            'y': vectors_2d[:, 1]
        })
        
        fig = px.scatter(
            df_viz,
            x='x',
            y='y',
            color='Word',
            text='Word',
            title='词向量2D可视化 (随机示例)',
            size=[20, 20, 20]
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            plot_bgcolor='#16213e',
            paper_bgcolor='#1a1a2e',
            font_color='#e6e6e6',
            height=500
        )
        st.plotly_chart(fig, width='stretch')
        
        st.info("💡 说明: 此演示使用随机向量。真实词向量需要在大规模语料上训练。")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>© 2024 语义表示与对比分析系统 | 自然语言处理课程演示平台</p>
</div>
""", unsafe_allow_html=True)
