import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import math
import time
import re
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score

# ==========================================
# 1. 頁面設定與 Steam 深色電競風 CSS
# ==========================================
st.set_page_config(
    page_title="Steam 遊戲痛點與流失率評估系統 | Data Science Ultimate",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #1b2838 0%, #0e1117 45%, #0b0e14 100%);
    }
    .hero-banner {
        background: linear-gradient(90deg, rgba(23,26,33,0.96) 0%, rgba(42,71,94,0.88) 100%);
        border: 1px solid #2a475e;
        border-left: 6px solid #66c0f4;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(102, 192, 244, 0.16);
        color: #66c0f4;
        border: 1px solid rgba(102, 192, 244, 0.45);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11.5px;
        font-weight: 700;
        margin-bottom: 6px;
        margin-right: 6px;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .hero-sub {
        font-size: 13.5px;
        color: #c7d5e0;
    }
    .kpi-card {
        background: linear-gradient(145deg, #171a21 0%, #1e2532 100%);
        border: 1px solid #2a3f5a;
        border-radius: 10px;
        padding: 15px 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.28);
    }
    .kpi-title { font-size: 12.5px; color: #8f98a0; margin-bottom: 5px; font-weight: 600; }
    .kpi-value { font-size: 26px; font-weight: 800; margin-bottom: 3px; }
    .kpi-desc { font-size: 11.5px; color: #66c0f4; }
    .review-card {
        background-color: #16202d;
        border: 1px solid #2a475e;
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        line-height: 1.65;
        color: #e6edf3;
    }
    .review-meta { font-size: 12px; color: #66c0f4; font-weight: 700; margin-bottom: 6px; }
    .kw-highlight {
        background-color: rgba(255, 75, 75, 0.28);
        color: #ff8a8a;
        padding: 1px 5px;
        border-radius: 4px;
        font-weight: bold;
        border-bottom: 1px solid #ff4b4b;
    }
    .action-card {
        background: #171d29;
        border: 1px solid #2c3e50;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div>
        <span class="hero-badge">東吳大學 資料科學系專題系統</span>
        <span class="hero-badge">Sentence-level ABSA</span>
        <span class="hero-badge">Bayesian & Chi-Square OR</span>
        <span class="hero-badge">ML Churn Prediction (AUC)</span>
    </div>
    <div class="hero-title">🎮 Steam 遊戲痛點與玩家流失率影響評估系統</div>
    <div class="hero-sub">整合自動化 ETL 採集、分句級屬性情感分析、貝葉斯平滑與卡方檢定（Odds Ratio 95% CI）、機器學習流失預測模型與動態敏感度驗證，打造端到端商業決策支援儀表板。</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 領域關鍵字詞典與基準樣本池
# ==========================================
ASPECT_KEYWORDS = {
    "💥 效能與 Bug": ["閃退", "崩潰", "優化", "幀數", "fps", "黑屏", "卡死", "bug", "報錯", "顯卡", "掉幀", "當機", "著色器", "卡頓", "記憶體"],
    "🔌 網路連線": ["連線", "伺服器", "斷線", "延遲", "ping", "封包", "掉線", "土豆", "連不上", "匹配不到", "網路", "回溯"],
    "🕹️ 遊戲機制": ["玩法", "機制", "外掛", "作弊", "舉報", "手感", "地圖", "空氣牆", "匹配", "隊友", "通行證", "無聊", "引導", "視角", "尋路"],
    "⚖️ 數值平衡": ["平衡", "削弱", "增強", "超模", "傷害", "武器", "英雄", "角色", "數值", "強度", "破格", "難度", "判定"],
    "🎨 畫面美術": ["畫質", "畫面", "美術", "建模", "光影", "貼圖", "特效", "解像度", "解析度", "風格", "風景", "音樂"]
}

POS_WORDS = ["神作", "優秀", "精緻", "流暢", "好玩", "推薦", "喜歡", "超美", "滿意", "不錯", "穩定", "良心", "歡樂", "震撼", "頂尖", "很棒", "很頂"]
NEG_WORDS = ["爛", "差", "垃圾", "失望", "噁心", "退費", "退款", "毀了", "無語", "勸退", "折磨", "坐牢", "沒救", "慢", "煩", "糟糕", "嚴重", "崩潰", "閃退", "失衡"]

CHINESE_GAME_MAP = {
    "黑神話": {"appid": "2358720", "name": "Black Myth: Wukong (黑神話：悟空)"},
    "悟空": {"appid": "2358720", "name": "Black Myth: Wukong (黑神話：悟空)"},
    "帕魯": {"appid": "1623730", "name": "Palworld (幻獸帕魯)"},
    "幻獸帕魯": {"appid": "1623730", "name": "Palworld (幻獸帕魯)"},
    "絕地求生": {"appid": "578080", "name": "PUBG: BATTLEGROUNDS (絕地求生)"},
    "吃雞": {"appid": "578080", "name": "PUBG: BATTLEGROUNDS (絕地求生)"},
    "艾爾登法環": {"appid": "1245620", "name": "ELDEN RING (艾爾登法環)"},
    "法環": {"appid": "1245620", "name": "ELDEN RING (艾爾登法環)"},
    "賽博朋克": {"appid": "1091500", "name": "Cyberpunk 2077 (電馭叛客 2077)"},
    "電馭叛客": {"appid": "1091500", "name": "Cyberpunk 2077 (電馭叛客 2077)"},
    "魔物獵人": {"appid": "2246340", "name": "Monster Hunter Wilds
