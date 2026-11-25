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
# 1. 頁面設定與 Steam 深色展覽風 CSS
# ==========================================
st.set_page_config(
    page_title="Steam 遊戲體檢儀｜找出真正害玩家退坑的原因",
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
        background: linear-gradient(100deg, rgba(23,26,33,0.97) 0%, rgba(36,64,88,0.90) 100%);
        border: 1px solid #2a475e;
        border-left: 6px solid #66c0f4;
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        letter-spacing: 0.4px;
    }
    .hero-sub {
        font-size: 15px;
        color: #d6e4f0;
        line-height: 1.6;
    }
    .step-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 14px;
    }
    .step-box {
        background: rgba(14, 17, 23, 0.65);
        border: 1px solid rgba(102, 192, 244, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
    }
    .step-num { font-size: 12px; font-weight: 800; color: #66c0f4; margin-bottom: 3px; }
    .step-text { font-size: 13px; color: #e6edf3; line-height: 1.45; }
    .picker-header {
        background: rgba(23, 26, 33, 0.9);
        border: 1px solid #2a475e;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .genre-pill {
        display: inline-block;
        background: rgba(78, 205, 196, 0.14);
        color: #4ecdc4;
        border: 1px solid rgba(78, 205, 196, 0.4);
        border-radius: 16px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .verdict-box {
        background: linear-gradient(90deg, rgba(255,75,75,0.16) 0%, rgba(23,26,33,0.95) 75%);
        border: 1px solid rgba(255,75,75,0.55);
        border-left: 6px solid #ff4b4b;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 18px 0;
    }
    .verdict-title { font-size: 18px; font-weight: 800; color: #ff8a8a; margin-bottom: 6px; }
    .verdict-body { font-size: 14.5px; color: #eef4fa; line-height: 1.65; }
    .kpi-card {
        background: linear-gradient(145deg, #171a21 0%, #1e2532 100%);
        border: 1px solid #2a3f5a;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.28);
    }
    .kpi-title { font-size: 13.5px; color: #c7d5e0; margin-bottom: 6px; font-weight: 700; }
    .kpi-value { font-size: 28px; font-weight: 800; margin-bottom: 4px; }
    .kpi-desc { font-size: 12px; color: #8fb8d6; }
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
    .review-meta { font-size: 12.5px; color: #66c0f4; font-weight: 700; margin-bottom: 6px; }
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
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
    }
    .explain-strip {
        background: rgba(102, 192, 244, 0.08);
        border: 1px solid rgba(102, 192, 244, 0.28);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 14px;
        font-size: 14px;
        color: #dce9f5;
        line-height: 1.6;
    }
    .dot-blue {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background-color: #66c0f4; margin-right: 7px;
    }
    .dot-red {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background-color: #ff4b4b; margin-right: 7px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Steam 遊戲體檢儀：找出「真正害玩家退坑」的致命傷</div>
    <div class="hero-sub">
        玩家給負評時，有些只是嘴上抱怨、隔天照樣玩；有些卻是<strong>罵完再也不開遊戲，甚至直接退款</strong>。<br>
        這套系統會自動讀取玩家評論與後續遊玩紀錄，幫開發團隊從海量留言中揪出<strong>最致命的退坑原因</strong>，並自動推薦<strong>同類型競品</strong>進行橫向比較。
    </div>
    <div class="step-grid">
        <div class="step-box">
            <div class="step-num">STEP 1｜抓評論＋看後續遊玩時數</div>
            <div class="step-text">比對「寫負評時」與「總遊玩時數」，分辨玩家是嘴上抱怨，還是真的退坑了。</div>
        </div>
        <div class="step-box">
            <div class="step-num">STEP 2｜分句拆解，不誤傷優點</div>
            <div class="step-text">把「畫面很美，但一直閃退」拆成子句分析，精準歸類五大痛點，不誤判好評。</div>
        </div>
        <div class="step-box">
            <div class="step-num">STEP 3｜算出 0～100 修復急迫度排行</div>
            <div class="step-text">結合統計檢定與機器學習，直接給出「馬上修 / 近期修 / 慢慢調」決策建議。</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 領域關鍵字詞典、同類型競品庫與基準樣本池
# ==========================================
ASPECT_KEYWORDS = {
    "效能與 Bug": ["閃退", "崩潰", "優化", "幀數", "fps", "黑屏", "卡死", "bug", "報錯", "顯卡", "掉幀", "當機", "著色器", "卡頓", "記憶體"],
    "網路連線": ["連線", "伺服器", "斷線", "延遲", "ping", "封包", "掉線", "土豆", "連不上", "匹配不到", "網路", "回溯"],
    "遊戲機制": ["玩法", "機制", "外掛", "作弊", "舉報", "手感", "地圖", "空氣牆", "匹配", "隊友", "通行證", "無聊", "引導", "視角", "尋路"],
    "數值平衡": ["平衡", "削弱", "增強", "超模", "傷害", "武器", "英雄", "角色", "數值", "強度", "破格", "難度", "判定"],
    "畫面美術": ["畫質", "畫面", "美術", "建模", "光影", "貼圖", "特效", "解像度", "解析度", "風格", "風景", "音樂"]
}

ASPECT_PLAIN_DESC = {
    "效能與 Bug": "閃退、黑屏、掉幀卡頓或遊戲當機",
    "網路連線": "伺服器斷線、高延遲跳 Ping 或連不上線",
    "遊戲機制": "外掛作弊、空氣牆、匹配失衡或玩法卡關",
    "數值平衡": "角色武器太強/太弱、後期難度失衡",
    "畫面美術": "畫質模糊、貼圖延遲載入或視覺表現"
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
    "隻狼": {"appid": "814380", "name": "Sekiro: Shadows Die Twice (隻狼：暗影雙死)"},
    "賽博朋克": {"appid": "1091500", "name": "Cyberpunk 2077 (電馭叛客 2077)"},
    "電馭叛客": {"appid": "1091500", "name": "Cyberpunk 2077 (電馭叛客 2077)"},
    "巫師": {"appid": "292030", "name": "The Witcher 3: Wild Hunt (巫師 3)"},
    "魔物獵人": {"appid": "2246340", "name": "Monster Hunter Wilds (魔物獵人 荒野)"},
    "柏德之門": {"appid": "1086940", "name": "Baldur's Gate 3 (柏德之門 3)"},
    "cs2": {"appid": "730", "name": "Counter-Strike 2 (CS2)"},
    "apex": {"appid": "1172470", "name": "Apex Legends"},
    "虹彩六號": {"appid": "359550", "name": "Rainbow Six Siege (虹彩六號：圍攻行動)"},
    "絕地戰兵": {"appid": "553850", "name": "HELLDIVERS 2 (絕地戰兵 2)"},
    "瓦爾海姆": {"appid": "892970", "name": "Valheim (瓦爾海姆)"},
    "方舟": {"appid": "2399830", "name": "ARK: Survival Ascended (方舟：生存飛升)"},
    "rust": {"appid": "252490", "name": "Rust"},
    "星露谷": {"appid": "413150", "name": "Stardew Valley (星露谷物語)"},
    "文明": {"appid": "289070", "name": "Civilization VI (文明帝國 VI)"},
    "都市天際線": {"appid": "949230", "name": "Cities: Skylines II (都市：天際線 2)"},
    "地平線": {"appid": "1551360", "name": "Forza Horizon 5 (極限競速：地平線 5)"},
    "2k": {"appid": "2878980", "name": "NBA 2K25"},
    "fc": {"appid": "2669320", "name": "EA SPORTS FC 25"}
}

# 同類型競品資料庫（供系統依據遊戲 A 的類型自動推薦同類遊戲）
SAME_GENRE_CATALOG = [
    {"appid": "2358720", "name": "Black Myth: Wukong (黑神話：悟空)", "short": "黑神話：悟空", "category": "動作 RPG", "genres": ["動作", "角色扮演", "冒險", "Action", "RPG", "Adventure"]},
    {"appid": "1245620", "name": "ELDEN RING (艾爾登法環)", "short": "艾爾登法環", "category": "動作 RPG", "genres": ["動作", "角色扮演", "Action", "RPG"]},
    {"appid": "814380", "name": "Sekiro: Shadows Die Twice (隻狼：暗影雙死)", "short": "隻狼：暗影雙死", "category": "動作 RPG", "genres": ["動作", "冒險", "Action", "Adventure"]},
    {"appid": "2246340", "name": "Monster Hunter Wilds (魔物獵人 荒野)", "short": "魔物獵人 荒野", "category": "動作 RPG", "genres": ["動作", "角色扮演", "冒險", "Action", "RPG", "Adventure"]},
    {"appid": "1091500", "name": "Cyberpunk 2077 (電馭叛客 2077)", "short": "電馭叛客 2077", "category": "開放世界 RPG", "genres": ["角色扮演", "動作", "RPG", "Action"]},
    {"appid": "292030", "name": "The Witcher 3: Wild Hunt (巫師 3)", "short": "巫師 3：狂獵", "category": "開放世界 RPG", "genres": ["角色扮演", "RPG"]},
    {"appid": "1086940", "name": "Baldur's Gate 3 (柏德之門 3)", "short": "柏德之門 3", "category": "角色扮演 RPG", "genres": ["角色扮演", "策略", "冒險", "RPG", "Strategy", "Adventure"]},
    {"appid": "578080", "name": "PUBG: BATTLEGROUNDS (絕地求生)", "short": "PUBG 絕地求生", "category": "多人射擊競技", "genres": ["動作", "冒險", "大型多人連線", "免費遊玩", "Action", "Massively Multiplayer", "Free To Play"]},
    {"appid": "1172470", "name": "Apex Legends", "short": "Apex Legends", "category": "多人射擊競技", "genres": ["動作", "冒險", "免費遊玩", "Action", "Free To Play"]},
    {"appid": "730", "name": "Counter-Strike 2 (CS2)", "short": "CS2", "category": "多人射擊競技", "genres": ["動作", "免費遊玩", "Action", "Free To Play"]},
    {"appid": "359550", "name": "Rainbow Six Siege (虹彩六號：圍攻行動)", "short": "虹彩六號：圍攻行動", "category": "多人射擊競技", "genres": ["動作", "Action"]},
    {"appid": "553850", "name": "HELLDIVERS 2 (絕地戰兵 2)", "short": "絕地戰兵 2", "category": "多人合作射擊", "genres": ["動作", "Action"]},
    {"appid": "1623730", "name": "Palworld (幻獸帕魯)", "short": "幻獸帕魯", "category": "開放世界生存", "genres": ["動作", "冒險", "獨立", "角色扮演", "搶先體驗", "Action", "Adventure", "Indie", "RPG"]},
    {"appid": "2399830", "name": "ARK: Survival Ascended (方舟：生存飛升)", "short": "方舟：生存飛升", "category": "開放世界生存", "genres": ["動作", "冒險", "獨立", "角色扮演", "大型多人連線", "Action", "Adventure", "Indie", "RPG"]},
    {"appid": "892970", "name": "Valheim (瓦爾海姆)", "short": "瓦爾海姆", "category": "開放世界生存", "genres": ["動作", "冒險", "獨立", "角色扮演", "Action", "Adventure", "Indie", "RPG"]},
    {"appid": "252490", "name": "Rust", "short": "Rust 腐蝕", "category": "多人開放世界生存", "genres": ["動作", "冒險", "獨立", "大型多人連線", "角色扮演", "Action", "Adventure", "Indie", "Massively Multiplayer"]},
    {"appid": "2878980", "name": "NBA 2K25", "short": "NBA 2K25", "category": "運動競技", "genres": ["運動", "模擬", "Sports", "Simulation"]},
    {"appid": "2669320", "name": "EA SPORTS FC 25", "short": "EA SPORTS FC 25", "category": "運動競技", "genres": ["運動", "模擬", "Sports", "Simulation"]},
    {"appid": "1551360", "name": "Forza Horizon 5 (極限競速：地平線 5)", "short": "極限競速：地平線 5", "category": "競速模擬", "genres": ["競速", "運動", "模擬", "動作", "Racing", "Sports", "Simulation"]},
    {"appid": "227300", "name": "Euro Truck Simulator 2 (歐洲卡車模擬 2)", "short": "歐洲卡車模擬 2", "category": "模擬經營", "genres": ["模擬", "獨立", "Simulation", "Indie"]},
    {"appid": "289070", "name": "Civilization VI (文明帝國 VI)", "short": "文明帝國 VI", "category": "策略模擬", "genres": ["策略", "Strategy"]},
    {"appid": "949230", "name": "Cities: Skylines II (都市：天際線 2)", "short": "都市：天際線 2", "category": "模擬經營", "genres": ["模擬", "策略", "Simulation", "Strategy"]},
    {"appid": "413150", "name": "Stardew Valley (星露谷物語)", "short": "星露谷物語", "category": "休閒模擬 RPG", "genres": ["獨立", "角色扮演", "模擬", "休閒", "Indie", "RPG", "Simulation", "Casual"]}
]

CURATED_PEER_MAP = {
    "2358720": ["1245620", "814380", "2246340"],
    "1245620": ["2358720", "814380", "2246340"],
    "814380":  ["2358720", "1245620", "2246340"],
    "2246340": ["2358720", "1245620", "814380"],
    "1091500": ["292030", "1245620", "1086940"],
    "578080":  ["1172470", "730", "359550"],
    "1172470": ["578080", "730", "359550"],
    "730":     ["578080", "1172470", "359550"],
    "359550":  ["730", "578080", "1172470"],
    "553850":  ["578080", "1172470", "359550"],
    "1623730": ["2399830", "892970", "252490"],
    "2399830": ["1623730", "892970", "252490"],
    "892970":  ["1623730", "2399830", "252490"],
    "252490":  ["1623730", "2399830", "892970"],
    "2878980": ["2669320", "1551360", "227300"],
    "2669320": ["2878980", "1551360", "227300"],
}

CATALOG_BY_ID = {g["appid"]: g for g in SAME_GENRE_CATALOG}

GAME_SNAPSHOT_POOLS = {
    "2358720": [
        ("第三章浮屠界掉幀超級嚴重，4070顯卡開中畫質還是狂閃退，著色器編譯崩潰三次直接退費", False, 1.5, 0.0, 48),
        ("剛買進去編譯著色器就一直黑屏閃退，貼圖載入也出問題，根本進不去遊戲已申請退款", False, 0.8, 0.0, 62),
        ("打完亢金龍直接黑屏閃退存檔差點壞掉，優化沒做好之前不建議入手", False, 8.4, 0.0, 19),
        ("畫面美術很頂，但水簾洞幀數暴跌到20幀還當機兩次，完全影響戰鬥體驗直接關遊戲", False, 1.9, 5.0, 34),
        ("雖然偶爾掉幀小卡頓但調低陰影後勉強能玩，希望官方盡快出優化補丁", False, 16.0, 480.0, 7),
        ("空氣牆多到離譜！地圖沒有小地圖引導很容易迷路，探索機制設計得很折磨人", False, 12.5, 10.0, 85),
        ("模組卡死在石頭縫裡兩次，加上視角鎖定bug，機制細節很粗糙勸退", False, 6.2, 20.0, 14),
        ("後期數值平衡有點失控，部分法術削弱後容錯率太低，玩到後面變成純坐牢", False, 31.5, 30.0, 27),
        ("某些Boss傷害數值設計不合理，但多死幾次背板還是打過了，希望後續調整平衡", False, 28.0, 650.0, 9),
        ("部分場景貼圖偶爾延遲載入，但整體畫面美術還是很頂，繼續推進度", False, 19.5, 900.0, 4),
        ("美術風格與戰鬥打擊感神作等級，雖然偶爾有小bug但整體非常震撼推薦", True, 45.2, 600.0, 31),
        ("畫質精緻到像在看電影，Boss戰演出超美，這價格完全良心之作", True, 52.0, 840.0, 40)
    ],
    "1245620": [
        ("DLC幽影樹數值平衡完全失控，王的傷害破格兩刀秒人，玩到後面變成純坐牢直接棄坑", False, 95.0, 0.0, 88),
        ("後期Boss難度曲線太離譜，武器削弱太多導致流派沒辦法玩，數值設計很讓人失望", False, 120.0, 10.0, 65),
        ("打大王時視角鎖定一直亂飄，地圖引導也不夠清楚，探索機制有時很折磨", False, 42.0, 15.0, 44),
        ("開放世界場景一開光追幀數就掉到30幀，偶爾還會卡頓閃退，優化還是不夠好", False, 1.8, 0.0, 37),
        ("連線合作找朋友召喚常常斷線連不上伺服器，多人機制真的該改進", False, 64.0, 25.0, 29),
        ("部分法術被削弱後打起來比較辛苦，但熟悉節奏後還是每天繼續推圖", False, 150.0, 520.0, 14),
        ("雖然偶爾卡頓但開放世界美術風景跟探索感依然神作等級，非常推薦", True, 180.0, 900.0, 52),
        ("美術光影與地圖設計頂尖，戰鬥手感流暢好玩，值得一玩再玩", True, 210.0, 1100.0, 41)
    ],
    "578080": [
        ("幀數優化還是一樣糟糕，煙霧彈一丟FPS直接掉到30，顯卡風扇狂轉直接關遊戲退坑", False, 1.7, 0.0, 42),
        ("每次改版更新完就必閃退報錯，記憶體占用高到離譜，優化越做越回去", False, 14.3, 0.0, 35),
        ("落地建築物貼圖還沒載入完成就卡死當機，畫面效能表現真的勸退", False, 1.9, 0.0, 28),
        ("亞服外掛滿天飛，每場決賽圈都有透視鎖頭，舉報系統根本裝飾用，徹底退坑", False, 420.0, 0.0, 112),
        ("匹配機制完全失衡加上伺服器延遲超高，新手剛玩就被炸魚還斷線，體驗極差", False, 9.5, 10.0, 49),
        ("更新後伺服器延遲超高，封包遺失率一直跳紅字，開鏡瞬間直接斷線掉線", False, 185.0, 12.0, 54),
        ("伺服器晚上尖峰時段偶爾會跳ping，只能換日服繼續跟朋友排", False, 310.0, 540.0, 11),
        ("武器數值平衡越改越爛，某幾把衝鋒槍傷害超模，但習慣後還是每天上線打兩把", False, 510.5, 420.0, 18),
        ("跟朋友四排吃雞還是很歡樂，地圖機制豐富，只是偶爾會遇到外掛", True, 650.0, 960.0, 22),
        ("槍械手感跟戰術配合依然是射擊遊戲頂尖，老玩家還是很推薦", True, 820.0, 1200.0, 19)
    ],
    "1172470": [
        ("伺服器根本土豆做的，每兩場就封包遺失掉線斷線，延遲高到沒辦法對槍直接退坑", False, 240.0, 0.0, 96),
        ("配對匹配機制極度失衡，單排一直排到頂獵三排車隊，打起來純坐牢勸退", False, 110.0, 5.0, 82),
        ("亞服外掛跟腳本作弊抓不完，舉報系統完全沒反應，徹底對這遊戲失望", False, 350.0, 0.0, 105),
        ("新賽季英雄數值平衡做壞了，某角色技能超模破壞體驗，希望快點削弱", False, 190.0, 420.0, 31),
        ("改版通行證機制越來越坑，加上偶爾伺服器跳ping，先退坑觀望", False, 85.0, 10.0, 48),
        ("更新後進遊戲偶爾報錯閃退，還好重開能連回去繼續打", False, 130.0, 480.0, 12),
        ("移動手感跟團隊槍戰節奏還是射擊遊戲裡最好玩的，跟朋友開黑很歡樂推薦", True, 410.0, 860.0, 39),
        ("角色技能搭配機制優秀，畫面特效流暢，老玩家依舊推薦", True, 520.0, 950.0, 27)
    ],
    "1623730": [
        ("記憶體洩漏問題嚴重，玩兩小時FPS就暴跌必須重開遊戲，優化太差退坑了", False, 1.9, 0.0, 43),
        ("據點帕魯一多幀數直接掉到個位數，還頻繁閃退回桌面，暫時不玩了", False, 15.4, 0.0, 31),
        ("官方伺服器卡頓到沒辦法玩，每十分鐘斷線回溯一次，蓋好的基地直接消失", False, 28.0, 0.0, 76),
        ("連線延遲嚴重且頻繁掉線，加上帕魯卡死bug，跟朋友連機體驗極差直接退坑", False, 11.2, 5.0, 51),
        ("帕魯在基地經常卡死在屋頂不吃飯，AI尋路機制bug太多，暫時先放著等更新", False, 42.5, 35.0, 21),
        ("後期抓寵數值平衡有點單調，但跟朋友連線蓋基地還是玩了很久", False, 55.0, 600.0, 8),
        ("玩法融合抓寵跟生存建造非常上頭，畫面風格可愛又好玩，非常推薦", True, 68.0, 780.0, 35)
    ]
}

DEFAULT_SNAPSHOT_POOL = [
    ("更新後一直閃退回桌面，剛買不到兩小時直接申請退款", False, 1.2, 0.0, 62),
    ("優化太差了，一進場就瘋狂閃退，根本沒辦法正常遊玩", False, 1.6, 0.0, 48),
    ("伺服器極度不穩定，打到一半突然斷線，這品質令人失望", False, 12.4, 5.0, 51),
    ("開中等設定還是狂掉幀，幀數暴跌超嚴重，希望快點修優化", False, 3.5, 0.0, 35),
    ("對戰環境外掛太多了，舉報系統根本沒用，徹底勸退", False, 58.0, 15.0, 74),
    ("晚上尖峰時段延遲超高，一直跳ping根本連不上伺服器", False, 18.0, 10.0, 33),
    ("進遊戲黑屏卡死兩次，連教學關都過不去直接退費", False, 0.8, 0.0, 44),
    ("匹配機制完全失衡，新手一直排到老手，體驗很差", False, 9.5, 10.0, 29),
    ("著色器編譯異常，顯卡風扇狂轉還當機一次，太折磨了", False, 2.2, 0.0, 26),
    ("武器與角色數值平衡糟糕，新角強度破格，希望下版削弱", False, 41.0, 360.0, 16),
    ("玩法後期重複度略高有點無聊，偶爾上線解每日任務還可以", False, 27.5, 420.0, 9),
    ("部分場景貼圖載入偏慢，但整體美術風格還是很有水準", False, 19.0, 480.0, 6),
    ("畫面美術與光影建模非常精緻，內容豐富很值得推薦", True, 55.0, 540.0, 24),
    ("跟朋友連線非常歡樂，整體手感流暢好玩，給個好評", True, 48.0, 720.0, 21)
]

# ==========================================
# 3. 核心演算法、Steam 遊戲類型偵測與同類推薦
# ==========================================
def classify_lifecycle(hrs):
    if hrs <= 2.0:
        return "1. 剛買就退坑 (≤2小時退款期)"
    elif hrs <= 20.0:
        return "2. 玩沒幾天棄坑 (2~20小時)"
    return "3. 老玩家心寒 (>20小時)"

def calc_odds_ratio_ci_pvalue(a, b, c, d):
    ap, bp, cp, dp = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (ap * dp) / (bp * cp)
    se_log_or = math.sqrt((1.0 / ap) + (1.0 / bp) + (1.0 / cp) + (1.0 / dp))
    ci_low = math.exp(math.log(or_val) - 1.96 * se_log_or)
    ci_high = math.exp(math.log(or_val) + 1.96 * se_log_or)

    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom <= 0 or n <= 0:
        return round(or_val, 2), round(ci_low, 2), round(ci_high, 2), 1.0
    chi2 = (n * ((a * d - b * c) ** 2)) / denom
    p_val = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return round(or_val, 2), round(ci_low, 2), round(ci_high, 2), p_val

def highlight_keywords(text, aspect_name):
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    kws = ASPECT_KEYWORDS.get(aspect_name, []) + NEG_WORDS
    for kw in sorted(set(kws), key=len, reverse=True):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f'<span class="kw-highlight">{m.group(0)}</span>', escaped)
    return escaped

@st.cache_data(ttl=3600)
def get_game_details(appid):
    """取得 Steam 遊戲的最新封面圖與官方遊戲類型標籤"""
    if not str(appid).isdigit():
        return {"header_image": None, "genres": []}
    headers = {"User-Agent": "Mozilla/5.0"}
    header_url = None
    genres = []
    try:
        res = requests.get(
            f"https://store.steampowered.com/api/appdetails?appids={appid}&l=tchinese",
            headers=headers, timeout=5
        )
        if res.status_code == 200:
            clean_txt = res.text.lstrip('\ufeff').strip()
            data = json.loads(clean_txt).get(str(appid), {})
            if data.get("success") and data.get("data"):
                info = data["data"]
                header_url = info.get("header_image")
                genres = [g.get("description", "") for g in info.get("genres", []) if g.get("description")]
    except Exception:
        pass

    if not genres and str(appid) in CATALOG_BY_ID:
        genres = [g for g in CATALOG_BY_ID[str(appid)]["genres"] if re.search(r'[\u4e00-\u9fff]', g)]

    if not header_url:
        for url in [
            f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg",
            f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
        ]:
            try:
                if requests.head(url, headers=headers, timeout=3).status_code == 200:
                    header_url = url
                    break
            except Exception:
                pass

    return {"header_image": header_url, "genres": genres}

def get_game_header_url(appid):
    return get_game_details(appid).get("header_image")

def recommend_same_genre_games(appid, game_name="", top_k=3):
    """依據目標遊戲的 Steam 類型標籤，自動推薦最相近的同類型競品"""
    aid = str(appid) if appid else ""
    details = get_game_details(aid) if aid.isdigit() else {"genres": []}
    target_genres = details.get("genres", [])

    if aid in CURATED_PEER_MAP:
        recs = [CATALOG_BY_ID[pid] for pid in CURATED_PEER_MAP[aid] if pid in CATALOG_BY_ID and pid != aid]
        return recs[:top_k], target_genres

    name_low = str(game_name).lower()
    inferred_genres = list(target_genres)
    if any(k in name_low for k in ["nba", "2k", "fc ", "fifa", "madden", "mlb", "wwe", "tennis"]):
        inferred_genres += ["運動", "模擬", "Sports", "Simulation"]
    elif any(k in name_low for k in ["forza", "f1", "need for speed", "racing", "rally", "競速"]):
        inferred_genres += ["競速", "運動", "Racing", "Sports"]
    elif any(k in name_low for k in ["call of duty", "battlefield", "valorant", "counter-strike", "shooter", "pubg", "apex"]):
        inferred_genres += ["動作", "大型多人連線", "免費遊玩", "Action", "Free To Play"]
    elif any(k in name_low for k in ["survival", "craft", "生存", "建造", "方舟", "帕魯"]):
        inferred_genres += ["冒險", "獨立", "角色扮演", "Adventure", "Indie", "RPG"]
    elif any(k in name_low for k in ["civilization", "total war", "age of empires", "cities", "策略", "模擬"]):
        inferred_genres += ["策略", "模擬", "Strategy", "Simulation"]

    target_set = set(g.lower() for g in inferred_genres if g)
    scored = []
    for item in SAME_GENRE_CATALOG:
        if item["appid"] == aid:
            continue
        item_set = set(g.lower() for g in item["genres"])
        overlap = len(target_set & item_set)
        bonus = 0
        for special in ["運動", "sports", "競速", "racing", "策略", "strategy", "模擬", "simulation", "免費遊玩", "free to play", "獨立", "indie"]:
            if special in target_set and special in item_set:
                bonus += 2
        scored.append((overlap + bonus, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        recs = [x[1] for x in scored[:top_k]]
    else:
        recs = [CATALOG_BY_ID["1245620"], CATALOG_BY_ID["814380"], CATALOG_BY_ID["2246340"]]
    return recs, (target_genres if target_genres else [g for g in inferred_genres if re.search(r'[\u4e00-\u9fff]', g)][:3])

@st.cache_data(ttl=600)
def smart_steam_search(term):
    if not term or not term.strip():
        return []
    term_clean = term.strip().lower()
    results, seen_ids = [], set()
    for zh_kw, info in CHINESE_GAME_MAP.items():
        if zh_kw in term_clean or term_clean in zh_kw:
            appid = info["appid"]
            if appid not in seen_ids:
                results.append({"appid": appid, "name": info["name"], "display": f"{info['name']} ({appid})"})
                seen_ids.add(appid)
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={term_clean}&l=zh-tw&cc=TW"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if res.status_code == 200:
            clean_txt = res.text.lstrip('\ufeff').strip()
            for item in json.loads(clean_txt).get("items", []):
                appid, name = str(item.get("id")), str(item.get("name"))
                if appid not in seen_ids:
                    results.append({"appid": appid, "name": name, "display": f"{name} ({appid})"})
                    seen_ids.add(appid)
    except Exception:
        pass
    return results

def fetch_steam_data(appid, target_count, churn_threshold_mins, progress_bar=None, status_text=None):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
    reviews_dict = {}
    for lang in ["tchinese", "schinese"]:
        if len(reviews_dict) >= target_count:
            break
        cursor, retry_fail = "*", 0
        while len(reviews_dict) < target_count:
            params = {
                "json": 1, "filter": "recent", "language": lang,
                "review_type": "all", "purchase_type": "all",
                "num_per_page": 100, "cursor": cursor
            }
            success, reviews, res_obj = False, [], {}
            for _ in range(2):
                try:
                    resp = requests.get(f"https://store.steampowered.com/appreviews/{appid}", params=params, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        clean_body = resp.text.lstrip('\ufeff').strip()
                        if clean_body.startswith("{"):
                            res_obj = json.loads(clean_body)
                            reviews = res_obj.get("reviews", [])
                            success = True
                            break
                except Exception:
                    time.sleep(0.5)
            if not success or not reviews:
                retry_fail += 1
                if retry_fail >= 1:
                    break
                continue
            for r in reviews:
                rid = str(r.get("recommendationid"))
                if rid not in reviews_dict:
                    author = r.get("author", {})
                    pt_forever = float(author.get("playtime_forever", 0))
                    pt_at_review = float(author.get("playtime_at_review", pt_forever))
                    post_mins = max(0.0, pt_forever - pt_at_review)
                    voted_up = bool(r.get("voted_up", False))
                    reviews_dict[rid] = {
                        "review_id": rid,
                        "text": str(r.get("review", "")),
                        "voted_up": voted_up,
                        "timestamp_created": r.get("timestamp_created"),
                        "playtime_at_review_hrs": round(pt_at_review / 60.0, 1),
                        "post_review_mins": post_mins,
                        "votes_up": int(r.get("votes_up", 0)),
                        "is_churned": (not voted_up) and (post_mins <= churn_threshold_mins)
                    }
            new_cursor = res_obj.get("cursor")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor
            if progress_bar and status_text:
                curr = min(len(reviews_dict), target_count)
                progress_bar.progress(curr / target_count)
                status_text.text(f"正在收集 Steam 玩家評論與遊玩時數... ({curr:,} / {target_count:,} 則)")
            time.sleep(0.25)

    df = pd.DataFrame(list(reviews_dict.values()))
    source_type = "live"

    if df.empty:
        source_type = "cached_benchmark"
        rng = random.Random(int(appid) if str(appid).isdigit() else 42)
        pool = GAME_SNAPSHOT_POOLS.get(str(appid), DEFAULT_SNAPSHOT_POOL)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=14)
        demo_rows = []
        weights = [max(3.0, (len(pool) - idx) ** 1.15) for idx in range(len(pool))]
        if len(weights) >= 2:
            weights[-2], weights[-1] = 16.0, 14.0

        for i in range(target_count):
            txt, is_up, b_hrs, b_post, b_votes = rng.choices(pool, weights=weights, k=1)[0]
            j_hrs = max(0.3, round(b_hrs * rng.lognormvariate(0, 0.55), 1))
            j_votes = max(0, int(b_votes * rng.lognormvariate(0, 0.7) + rng.randint(0, 5)))
            if not is_up:
                if b_post <= churn_threshold_mins:
                    if rng.random() < 0.18:
                        j_post = round(rng.uniform(churn_threshold_mins + 30, 540.0), 1)
                    else:
                        j_post = max(0.0, round(b_post * rng.uniform(0.3, 1.8) + rng.uniform(0, 15), 1))
                else:
                    if rng.random() < 0.22:
                        j_post = 0.0 if churn_threshold_mins <= 0 else round(rng.uniform(0.0, churn_threshold_mins * 0.8), 1)
                    else:
                        j_post = max(0.0, round(b_post * rng.uniform(0.5, 1.6), 1))
            else:
                j_post = max(0.0, round(b_post * rng.uniform(0.5, 1.5), 1))

            d = rng.choice(dates)
            demo_rows.append({
                "review_id": f"snap_{appid}_{i}",
                "text": txt,
                "voted_up": is_up,
                "timestamp_created": int(d.timestamp()),
                "playtime_at_review_hrs": j_hrs,
                "post_review_mins": j_post,
                "votes_up": j_votes,
                "is_churned": (not is_up) and (j_post <= churn_threshold_mins)
            })
        df = pd.DataFrame(demo_rows)

    df["timestamp_created"] = pd.to_numeric(df["timestamp_created"], errors="coerce")
    df = df.dropna(subset=["timestamp_created"])
    df["date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce").dt.date
    df = df.dropna(subset=["date"]).head(target_count).reset_index(drop=True)
    return df, source_type

def run_sentence_absa(df):
    results = []
    keyword_hits = {}
    for idx, row in df.iterrows():
        raw_text = str(row["text"])
        overall_neg = not bool(row["voted_up"])
        is_churned = bool(row["is_churned"])
        votes_up = max(0, int(row.get("votes_up", 0)))
        empathy_weight = 1.0 + np.log1p(votes_up)
        clauses = [c.strip().lower() for c in re.split(r'[，。！？；\n,.!?;]+', raw_text) if c.strip()]

        matched_aspects = {}
        for clause in clauses:
            has_pos = any(pw in clause for pw in POS_WORDS)
            has_neg = any(nw in clause for nw in NEG_WORDS)
            for aspect, keywords in ASPECT_KEYWORDS.items():
                hit_kws = [kw for kw in keywords if kw.lower() in clause]
                if hit_kws:
                    clause_is_neg = False if (has_pos and not has_neg) else (True if has_neg else overall_neg)
                    if clause_is_neg and is_churned:
                        for hk in hit_kws:
                            keyword_hits[hk] = keyword_hits.get(hk, 0) + 1
                    matched_aspects[aspect] = matched_aspects.get(aspect, False) or clause_is_neg

        for aspect, is_neg in matched_aspects.items():
            results.append({
                "review_idx": idx,
                "date": row["date"],
                "aspect": aspect,
                "is_neg": is_neg,
                "is_churned_pain": is_neg and is_churned,
                "playtime_hrs": float(row["playtime_at_review_hrs"]),
                "lifecycle": classify_lifecycle(float(row["playtime_at_review_hrs"])),
                "votes_up": votes_up,
                "empathy_weight": empathy_weight,
                "text": raw_text.strip()
            })
    return pd.DataFrame(results), keyword_hits

def calculate_ppi(df_reviews, df_absa, prior_strength=10.0):
    if df_absa.empty:
        return pd.DataFrame()
    total_weighted = df_absa["empathy_weight"].sum()
    g_neg = df_absa["is_neg"].mean()
    neg_only = df_absa[df_absa["is_neg"]]
    g_churn = neg_only["is_churned_pain"].mean() if not neg_only.empty else 0.0

    tot_n = len(df_reviews)
    tot_churn = int(df_reviews["is_churned"].sum())

    rows = []
    for asp in ASPECT_KEYWORDS.keys():
        sub = df_absa[df_absa["aspect"] == asp]
        if sub.empty:
            continue
        m_cnt = len(sub)
        n_cnt = int(sub["is_neg"].sum())
        c_cnt = int(sub["is_churned_pain"].sum())
        w_vol = sub["empathy_weight"].sum()

        a, b = c_cnt, max(0, n_cnt - c_cnt)
        c = max(0, tot_churn - a)
        d = max(0, (tot_n - n_cnt) - c)
        or_val, ci_low, ci_high, p_val = calc_odds_ratio_ci_pvalue(a, b, c, d)
        sig = f"{p_val:.3f} ***" if p_val < 0.001 else (f"{p_val:.3f} **" if p_val < 0.01 else (f"{p_val:.3f} *" if p_val < 0.05 else f"{p_val:.3f} (ns)"))

        raw_neg_rate = n_cnt / m_cnt if m_cnt > 0 else 0.0
        s_neg = (n_cnt + prior_strength * g_neg) / (m_cnt + prior_strength)
        s_churn = (c_cnt + prior_strength * g_churn) / (n_cnt + prior_strength)
        w_share = w_vol / total_weighted if total_weighted > 0 else 0.0
        med_pt = sub[sub["is_neg"]]["playtime_hrs"].median() if n_cnt > 0 else 0.0

        rows.append({
            "痛點屬性": asp,
            "提及次數": m_cnt,
            "負面提及數": n_cnt,
            "原始負評率 (%)": round(raw_neg_rate * 100, 1),
            "平滑負評率 (%)": round(s_neg * 100, 1),
            "平滑流失轉化率 (%)": round(s_churn * 100, 1),
            "流失勝算比 (OR)": f"{or_val}x",
            "OR 95% 信賴區間": f"[{ci_low}, {ci_high}]",
            "卡方檢定 p-value": sig,
            "負評中位數時數 (hr)": round(med_pt, 1),
            "_or_num": or_val,
            "_ci_low": ci_low,
            "_ci_high": ci_high,
            "_raw": (w_share * 0.30) + (s_neg * 0.35) + (s_churn * 0.35)
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    max_s = res["_raw"].max()
    res["痛點衝擊指數 (PPI)"] = np.round((res["_raw"] / max_s) * 100, 1) if max_s > 0 else 0.0
    return res.sort_values("痛點衝擊指數 (PPI)", ascending=False).reset_index(drop=True)

def build_ml_dataset(df_reviews, df_absa):
    feat = pd.DataFrame(index=df_reviews.index)
    feat["遊玩時數 (對數)"] = np.log1p(df_reviews["playtime_at_review_hrs"].clip(lower=0))
    feat["他人按讚共鳴 (對數)"] = np.log1p(df_reviews["votes_up"].clip(lower=0))
    feat["評論字數長度"] = df_reviews["text"].astype(str).str.len()

    neg_absa = df_absa[df_absa["is_neg"]] if not df_absa.empty else pd.DataFrame(columns=["review_idx", "aspect"])
    for asp in ASPECT_KEYWORDS.keys():
        hit_idxs = set(neg_absa[neg_absa["aspect"] == asp]["review_idx"].tolist())
        feat[f"遇到痛點：{asp}"] = [1.0 if i in hit_idxs else 0.0 for i in df_reviews.index]

    y = df_reviews["is_churned"].astype(int).values
    return feat.fillna(0.0), y

def run_sensitivity_kendall_w(df_reviews, prior_strength):
    thresholds = [15, 30, 60, 90, 120, 180]
    rank_matrix = []
    trend_records = []
    aspects_order = list(ASPECT_KEYWORDS.keys())

    for t in thresholds:
        tmp_df = df_reviews.copy()
        tmp_df["is_churned"] = (~tmp_df["voted_up"].astype(bool)) & (tmp_df["post_review_mins"] <= t)
        tmp_absa, _ = run_sentence_absa(tmp_df)
        tmp_ppi = calculate_ppi(tmp_df, tmp_absa, prior_strength=prior_strength)
        if tmp_ppi.empty:
            continue
        ppi_map = dict(zip(tmp_ppi["痛點屬性"], tmp_ppi["痛點衝擊指數 (PPI)"]))
        scores = [ppi_map.get(a, 0.0) for a in aspects_order]
        ranks = pd.Series(scores).rank(ascending=False, method="average").values
        rank_matrix.append(ranks)
        for a, sc, rk in zip(aspects_order, scores, ranks):
            trend_records.append({"門檻 ΔT (分鐘)": f"≤ {t}m", "痛點屬性": a, "PPI 分數": sc, "急迫度排名": rk})

    if len(rank_matrix) < 2:
        return pd.DataFrame(trend_records), 1.0
    mat = np.array(rank_matrix)
    m, k = mat.shape
    rank_sums = mat.sum(axis=0)
    mean_r = m * (k + 1) / 2.0
    s_val = np.sum((rank_sums - mean_r) ** 2)
    w_val = (12.0 * s_val) / (m ** 2 * (k ** 3 - k)) if k > 1 else 1.0
    return pd.DataFrame(trend_records), round(min(1.0, max(0.0, w_val)), 4)

def generate_markdown_brief(game_name, total_n, neg_ratio, churn_in_neg, refund_cnt, churn_mins, ppi_df):
    top1 = ppi_df.iloc[0]
    cols = ["痛點屬性", "提及次數", "負面提及數", "平滑負評率 (%)", "平滑流失轉化率 (%)", "流失勝算比 (OR)", "卡方檢定 p-value", "痛點衝擊指數 (PPI)"]
    sub_tbl = ppi_df[cols]
    try:
        md_table = sub_tbl.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub_tbl.values]
        md_table = "\n".join([header, sep] + body)

    lines = [
        f"# Steam 遊戲體檢報告：{game_name}",
        "",
        "## 1. 核心指標摘要",
        f"- **分析評論總數**：{total_n:,} 則",
        f"- **整體負評比例**：{neg_ratio:.1f}%",
        f"- **罵完真的退坑比例 (評論後遊玩 ≤ {churn_mins} 分鐘)**：{churn_in_neg:.1f}%",
        f"- **2 小時內退款高風險人數**：{refund_cnt} 人",
        "",
        "## 2. 痛點修復急迫度與統計檢定表",
        md_table,
        "",
        "## 3. 優先修復建議",
        f"- **第一優先（緊急熱修）**：【{top1['痛點屬性']}】（急迫度 {top1['痛點衝擊指數 (PPI)']} 分｜退坑風險倍數 {top1['流失勝算比 (OR)']}｜中位數遊玩時數 {top1['負評中位數時數 (hr)']} 小時）。"
    ]
    return "\n".join(lines)

# ==========================================
# 4. Session State 初始化與共用執行函式
# ==========================================
for key, default in [
    ("data_ready", False), ("df_reviews", None), ("source_type", ""),
    ("selected_name", ""), ("selected_appid", ""),
    ("pk_df", None), ("pk_name", ""), ("pk_appid", ""), ("pk_enabled", True)
]:
    if key not in st.session_state:
        st.session_state[key] = default

def load_game_pair(main_appid, main_name, pk_appid, pk_name, n_cnt, c_mins, use_pk=True):
    p_bar = st.progress(0)
    s_txt = st.empty()
    df_main, src_type = fetch_steam_data(main_appid, n_cnt, c_mins, p_bar, s_txt)
    p_bar.empty()
    s_txt.empty()
    st.session_state["df_reviews"] = df_main
    st.session_state["source_type"] = src_type
    st.session_state["selected_name"] = main_name
    st.session_state["selected_appid"] = main_appid
    st.session_state["pk_enabled"] = bool(use_pk and pk_appid)
    if use_pk and pk_appid:
        df_comp, _ = fetch_steam_data(pk_appid, n_cnt, c_mins)
        st.session_state["pk_df"] = df_comp
        st.session_state["pk_name"] = pk_name
        st.session_state["pk_appid"] = pk_appid
    else:
        st.session_state["pk_df"] = None
        st.session_state["pk_name"] = ""
        st.session_state["pk_appid"] = ""
    st.session_state["data_ready"] = True

# ==========================================
# 5. 側邊欄：隨時切換遊戲 ＋ 進階資料科學參數
# ==========================================
with st.sidebar:
    st.header("操作面板")
    if st.session_state["data_ready"]:
        if st.button("回到首頁重新挑選遊戲", use_container_width=True):
            st.session_state["data_ready"] = False
            st.rerun()
        st.divider()

    st.subheader("快速切換／比較遊戲")
    sb_kw1 = st.text_input("主體檢遊戲", value="黑神話", key="sb_kw1")
    sb_games1 = smart_steam_search(sb_kw1)
    sb_appid1, sb_name1 = None, ""
    if sb_games1:
        sb_opts1 = {g["display"]: g for g in sb_games1}
        sb_ch1 = st.selectbox("選擇主遊戲", list(sb_opts1.keys()), key="sb_sel1")
        sb_appid1, sb_name1 = sb_opts1[sb_ch1]["appid"], sb_opts1[sb_ch1]["name"]
    elif sb_kw1.isdigit():
        sb_appid1, sb_name1 = sb_kw1, f"AppID {sb_kw1}"

    sb_pk_on = st.toggle("開啟雙遊戲 PK 比較", value=True, key="sb_pk_on")
    sb_appid2, sb_name2 = None, ""
    if sb_pk_on:
        sb_kw2 = st.text_input("對照組遊戲", value="艾爾登法環", key="sb_kw2")
        sb_games2 = smart_steam_search(sb_kw2)
        if sb_games2:
            sb_opts2 = {g["display"]: g for g in sb_games2}
            sb_ch2 = st.selectbox("選擇對照遊戲", list(sb_opts2.keys()), key="sb_sel2")
            sb_appid2, sb_name2 = sb_opts2[sb_ch2]["appid"], sb_opts2[sb_ch2]["name"]
        elif sb_kw2.isdigit():
            sb_appid2, sb_name2 = sb_kw2, f"AppID {sb_kw2}"

    sb_run = st.button("套用並開始體檢", type="primary", use_container_width=True, key="sb_run")

    st.divider()
    with st.expander("進階資料科學參數（評審/研究用）", expanded=False):
        target_count = st.slider("分析評論數量 (N)", 100, 1200, 400, 100)
        churn_mins = st.slider("判定「真的退坑」門檻 ΔT (分鐘)", 0, 180, 60, 15,
                               help="留負評後，如果後續新增遊玩時間不超過這個分鐘數，就判定為實質退坑")
        prior_c = st.slider("貝葉斯小樣本平滑強度 (C)", 0, 20, 10, 2,
                            help="防止冷門屬性因為樣本太少而衝到 100% 負評")
        uploaded_csv = st.file_uploader("上傳本地評論 CSV 檔", type=["csv"])

if uploaded_csv is not None:
    df_up = pd.read_csv(uploaded_csv)
    if "votes_up" not in df_up.columns:
        df_up["votes_up"] = 0
    if "date" not in df_up.columns:
        df_up["date"] = pd.Timestamp.today().date()
    st.session_state["df_reviews"] = df_up
    st.session_state["source_type"] = "uploaded_csv"
    st.session_state["selected_name"] = f"自訂資料集 ({uploaded_csv.name})"
    st.session_state["selected_appid"] = "custom"
    st.session_state["pk_enabled"] = False
    st.session_state["data_ready"] = True

if sb_run and sb_appid1:
    load_game_pair(sb_appid1, sb_name1, sb_appid2, sb_name2, target_count, churn_mins, use_pk=sb_pk_on)

# ==========================================
# 6. 首頁畫面：自由選擇遊戲 ＋ 同類型競品自動推薦
# ==========================================
if not st.session_state["data_ready"]:
    st.markdown("""
    <div class="picker-header">
        <div style="font-size:19px;font-weight:800;color:#ffffff;margin-bottom:4px;">第一步：選擇你想體檢與比較的 Steam 遊戲</div>
        <div style="font-size:13.5px;color:#b8cde0;">先在左側選出「遊戲 A」，系統會自動偵測它的 Steam 遊戲類型，並在下方推薦<strong>同類型的競品遊戲</strong>讓你一鍵比較。</div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_vs, col_g2 = st.columns([1.2, 0.25, 1.2])

    with col_g1:
        st.markdown('<h4><span class="dot-blue"></span>遊戲 A（主要體檢對象）</h4>', unsafe_allow_html=True)
        home_kw1 = st.text_input("輸入遊戲名稱或 AppID", value="黑神話", key="home_kw1")
        home_games1 = smart_steam_search(home_kw1)
        h_appid1, h_name1 = None, ""
        if home_games1:
            h_opts1 = {g["display"]: g for g in home_games1}
            h_ch1 = st.selectbox("選擇主要遊戲", list(h_opts1.keys()), key="home_sel1")
            h_appid1, h_name1 = h_opts1[h_ch1]["appid"], h_opts1[h_ch1]["name"]
        elif home_kw1.isdigit():
            h_appid1, h_name1 = home_kw1, f"AppID {home_kw1}"

        if h_appid1:
            g1_details = get_game_details(h_appid1)
            if g1_details["genres"]:
                pills_html = "".join([f'<span class="genre-pill">{g}</span>' for g in g1_details["genres"][:4]])
                st.markdown(pills_html, unsafe_allow_html=True)
            if g1_details["header_image"]:
                st.image(g1_details["header_image"], use_container_width=True)

    with col_vs:
        st.markdown("<div style='text-align:center;font-size:32px;font-weight:900;color:#66c0f4;margin-top:95px;'>VS</div>", unsafe_allow_html=True)

    with col_g2:
        st.markdown('<h4><span class="dot-red"></span>遊戲 B（同類型比較對象）</h4>', unsafe_allow_html=True)
        home_pk_on = st.checkbox("同時比較第二款遊戲（開啟雷達圖 PK）", value=True, key="home_pk_on")
        h_appid2, h_name2 = None, ""
        if home_pk_on:
            home_kw2 = st.text_input("輸入要比較的遊戲名稱或 AppID", value="艾爾登法環", key="home_kw2")
            home_games2 = smart_steam_search(home_kw2)
            if home_games2:
                h_opts2 = {g["display"]: g for g in home_games2}
                h_ch2 = st.selectbox("選擇比較遊戲", list(h_opts2.keys()), key="home_sel2")
                h_appid2, h_name2 = h_opts2[h_ch2]["appid"], h_opts2[h_ch2]["name"]
            elif home_kw2.isdigit():
                h_appid2, h_name2 = home_kw2, f"AppID {home_kw2}"

            if h_appid2:
                g2_details = get_game_details(h_appid2)
                if g2_details["genres"]:
                    pills_html2 = "".join([f'<span class="genre-pill">{g}</span>' for g in g2_details["genres"][:4]])
                    st.markdown(pills_html2, unsafe_allow_html=True)
                if g2_details["header_image"]:
                    st.image(g2_details["header_image"], use_container_width=True)
        else:
            st.info("目前已切換為「單款遊戲深度體檢」模式，若想進行雙遊戲比較，勾選上方選項即可。")

    st.write("")
    btn_label = f"開始體檢並比較：{h_name1} vs. {h_name2}" if (home_pk_on and h_appid2) else f"開始體檢：{h_name1}"
    if st.button(btn_label, type="primary", use_container_width=True, key="home_start_btn"):
        if h_appid1:
            load_game_pair(h_appid1, h_name1, h_appid2, h_name2, target_count, churn_mins, use_pk=home_pk_on)
            st.rerun()
        else:
            st.warning("請先在左側輸入有效的 Steam 遊戲名稱或 AppID。")

    st.divider()
    recs, detected_genres = recommend_same_genre_games(h_appid1, h_name1, top_k=3)
    genre_label = " / ".join(detected_genres[:3]) if detected_genres else "同類型熱門"
    st.markdown(f"#### 為《{h_name1 or '目標遊戲'}》推薦的「同類型競品」一鍵比較")
    st.caption(f"系統已自動比對 Steam 官方遊戲類型（**{genre_label}**），點擊下方任一款同類對手即可直接進入雙遊戲雷達圖對比：")

    rec_cols = st.columns(3)
    for idx, peer in enumerate(recs):
        with rec_cols[idx]:
            short_a = CATALOG_BY_ID[str(h_appid1)]["short"] if str(h_appid1) in CATALOG_BY_ID else (h_name1[:16] if h_name1 else "遊戲 A")
            btn_text = f"{short_a} vs {peer['short']}\n（同屬：{peer['category']}）"
            if st.button(btn_text, use_container_width=True, key=f"rec_same_genre_{idx}_{peer['appid']}"):
                target_a_id = h_appid1 or "2358720"
                target_a_name = h_name1 or "Black Myth: Wukong (黑神話：悟空)"
                load_game_pair(target_a_id, target_a_name, peer["appid"], peer["name"], target_count, churn_mins, use_pk=True)
                st.rerun()

    st.write("")
    with st.expander("查看其他熱門「同類型」展覽示範組（射擊／生存／運動）", expanded=False):
        g_col1, g_col2, g_col3 = st.columns(3)
        if g_col1.button("PUBG vs Apex Legends\n（同屬：多人大逃殺射擊）", use_container_width=True, key="preset_fps"):
            load_game_pair("578080", "PUBG: BATTLEGROUNDS (絕地求生)", "1172470", "Apex Legends", target_count, churn_mins, use_pk=True)
            st.rerun()
        if g_col2.button("幻獸帕魯 vs 方舟：生存飛升\n（同屬：開放世界生存建造）", use_container_width=True, key="preset_surv"):
            load_game_pair("1623730", "Palworld (幻獸帕魯)", "2399830", "ARK: Survival Ascended (方舟：生存飛升)", target_count, churn_mins, use_pk=True)
            st.rerun()
        if g_col3.button("NBA 2K25 vs EA SPORTS FC 25\n（同屬：年度運動競技模擬）", use_container_width=True, key="preset_sport"):
            load_game_pair("2878980", "NBA 2K25", "2669320", "EA SPORTS FC 25", target_count, churn_mins, use_pk=True)
            st.rerun()

# ==========================================
# 7. 分析結果儀表板（4 大核心分頁）
# ==========================================
else:
    df_reviews = st.session_state["df_reviews"].copy()
    df_reviews["is_churned"] = (~df_reviews["voted_up"].astype(bool)) & (df_reviews["post_review_mins"] <= churn_mins)

    selected_name = st.session_state["selected_name"]
    selected_appid = st.session_state["selected_appid"]
    source_type = st.session_state["source_type"]

    df_absa, kw_hits = run_sentence_absa(df_reviews)
    ppi_df = calculate_ppi(df_reviews, df_absa, prior_strength=float(prior_c))

    total_n = len(df_reviews)
    neg_df = df_reviews[~df_reviews["voted_up"]]
    neg_ratio = (len(neg_df) / total_n) * 100 if total_n > 0 else 0.0
    churn_in_neg = (neg_df["is_churned"].sum() / len(neg_df) * 100) if len(neg_df) > 0 else 0.0
    refund_cnt = len(neg_df[(neg_df["is_churned"]) & (neg_df["playtime_at_review_hrs"] <= 2.0)])

    X_df, y_arr = build_ml_dataset(df_reviews, df_absa)
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    ml_ready = (len(unique_classes) == 2 and counts.min() >= 5 and len(X_df) >= 30)
    lr_pipe, rf_clf, X_te, y_te = None, None, None, None
    if ml_ready:
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_arr, test_size=0.25, random_state=42, stratify=y_arr)
        lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
        rf_clf = RandomForestClassifier(n_estimators=120, max_depth=5, class_weight="balanced", random_state=42)
        lr_pipe.fit(X_tr, y_tr)
        rf_clf.fit(X_tr, y_tr)

    col_img, col_info = st.columns([1.1, 2.9])
    with col_img:
        header_url = get_game_header_url(selected_appid)
        if header_url:
            st.image(header_url, use_container_width=True)
        else:
            st.info(selected_name)

    with col_info:
        top_r1, top_r2 = st.columns([2.6, 1.0])
        with top_r1:
            st.subheader(f"目前體檢遊戲：{selected_name}")
        with top_r2:
            if st.button("重選／更換比較遊戲", use_container_width=True):
                st.session_state["data_ready"] = False
                st.rerun()

        if source_type == "live":
            st.success("資料來源：已連線 Steam 官方商店，即時抓取最新繁／簡體中文評論完成分析")
        elif source_type == "uploaded_csv":
            st.success("資料來源：使用本地上傳的評論資料集完成分析")
        else:
            st.info("資料來源：已自動載入該遊戲離線歷史基準資料集（Offline Benchmark Snapshot）確保展覽不斷線")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("下載原始評論資料 (.csv)", data=df_reviews.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"steam_data_{selected_appid}.csv", mime="text/csv", use_container_width=True)
        with dl2:
            if not ppi_df.empty:
                md_brief = generate_markdown_brief(selected_name, total_n, neg_ratio, churn_in_neg, refund_cnt, churn_mins, ppi_df)
                st.download_button("匯出體檢報告 (.md)", data=md_brief.encode("utf-8"),
                                   file_name=f"steam_report_{selected_appid}.md", mime="text/markdown", use_container_width=True)

    if not ppi_df.empty:
        top1 = ppi_df.iloc[0]
        top1_name = top1["痛點屬性"]
        top1_plain = ASPECT_PLAIN_DESC.get(top1_name, "核心遊戲體驗問題")
        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-title">3 秒體檢結論：最害《{selected_name}》玩家退坑的頭號殺手是【{top1_name}】</div>
            <div class="verdict-body">
                • <strong>核心災情</strong>：大量玩家反映「<strong>{top1_plain}</strong>」，系統算出它的修復急迫度高達 <strong>{top1['痛點衝擊指數 (PPI)']} / 100 分</strong>（退坑風險是一般玩家的 <strong>{top1['流失勝算比 (OR)']}</strong>）。<br>
                • <strong>營收警訊</strong>：每 10 位給負評的玩家裡，就有 <strong>{churn_in_neg/10:.1f} 位</strong>罵完後幾乎沒再開過遊戲，其中 <strong>{refund_cnt} 位</strong>更是在剛買 2 小時內就踩到 Steam 無條件退款紅線。
            </div>
        </div>
        """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">有效分析評論</div><div class="kpi-value" style="color:#66c0f4;">{total_n:,} 則</div><div class="kpi-desc">已自動過濾重複與亂碼</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">整體不推薦比例</div><div class="kpi-value" style="color:#ffa15a;">{neg_ratio:.1f}%</div><div class="kpi-desc">共抓出 {len(neg_df):,} 則負面評論</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">罵完真的退坑比例</div><div class="kpi-value" style="color:#ff4b4b;">{churn_in_neg:.1f}%</div><div class="kpi-desc">留負評後幾乎沒再開過遊戲</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">2 小時內退款高風險</div><div class="kpi-value" style="color:#4ecdc4;">{refund_cnt} 人</div><div class="kpi-desc">剛買就退坑，直接損失營收</div></div>', unsafe_allow_html=True)

    st.write("")

    if df_absa.empty or ppi_df.empty:
        st.warning("目前樣本中未偵測到足夠的遊戲痛點關鍵字。")
    else:
        tab_labels = [
            "1. 一眼看懂：最該先修什麼",
            "2. 玩家什麼時候棄坑",
            "3. 同類型遊戲 PK",
            "4. 資料科學與統計驗證"
        ]
        tabs = st.tabs(tab_labels)

        # --- TAB 1: 一眼看懂 ---
        with tabs[0]:
            st.markdown("""
            <div class="explain-strip">
                <strong>閱讀指引：</strong> 左圖顯示哪個問題最容易逼走玩家（滿分 100 分）；右圖直接抓出退坑玩家最常抱怨的具體詞彙，下方附有給遊戲公司的修復順序建議與玩家真實留言。
            </div>
            """, unsafe_allow_html=True)

            c_left, c_right = st.columns([1.25, 1.0])
            with c_left:
                fig_ppi = px.bar(
                    ppi_df.sort_values("痛點衝擊指數 (PPI)", ascending=True),
                    x="痛點衝擊指數 (PPI)", y="痛點屬性", orientation="h",
                    text="痛點衝擊指數 (PPI)", color="痛點衝擊指數 (PPI)",
                    color_continuous_scale=["#4ecdc4", "#ffa15a", "#ff4b4b"],
                    template="plotly_dark", title="修復急迫度排行榜（分數越高＝越害玩家退坑）"
                )
                fig_ppi.update_traces(textposition="outside")
                fig_ppi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)",
                                      coloraxis_showscale=False, xaxis_range=[0, 118],
                                      xaxis_title="痛點衝擊指數 (滿分 100)", yaxis_title="", height=360)
                st.plotly_chart(fig_ppi, use_container_width=True)

            with c_right:
                if kw_hits:
                    kw_df = pd.DataFrame(list(kw_hits.items()), columns=["玩家常罵關鍵字", "退坑評論命中次數"]).sort_values("退坑評論命中次數", ascending=True).tail(8)
                    fig_kw = px.bar(kw_df, x="退坑評論命中次數", y="玩家常罵關鍵字", orientation="h", text="退坑評論命中次數",
                                    template="plotly_dark", title="退坑玩家最常提到的 8 個關鍵字")
                    fig_kw.update_traces(marker_color="#66c0f4", textposition="outside")
                    fig_kw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)",
                                         yaxis_title="", height=360)
                    st.plotly_chart(fig_kw, use_container_width=True)

            st.markdown("### 給遊戲公司的 3 階段修復建議")
            top1 = ppi_df.iloc[0]
            top2 = ppi_df.iloc[1] if len(ppi_df) > 1 else top1
            low_last = ppi_df.iloc[-1]

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; align-items: stretch;">
              <div class="action-card" style="border-top: 4px solid #ff4b4b; box-sizing: border-box;">
                <h4 style="color:#ff6b6b;">P0 第一優先：馬上緊急修復</h4>
                <p><strong>鎖定問題：{top1['痛點屬性']}</strong>（急迫度 {top1['痛點衝擊指數 (PPI)']} 分）</p>
                <p>遇到這個問題的玩家，中位數只玩了 <strong>{top1['負評中位數時數 (hr)']} 小時</strong>就寫負評，最容易引發剛買就退款的雪崩效應，建議研發團隊立刻優先搶修。</p>
              </div>
              <div class="action-card" style="border-top: 4px solid #ffa15a; box-sizing: border-box;">
                <h4 style="color:#ffa15a;">P1 第二優先：近期版本改善</h4>
                <p><strong>鎖定問題：{top2['痛點屬性']}</strong>（急迫度 {top2['痛點衝擊指數 (PPI)']} 分）</p>
                <p>這個問題會讓 <strong>{top2['平滑流失轉化率 (%)']}%</strong> 的負評玩家選擇退坑，主要傷害玩了約 <strong>{top2['負評中位數時數 (hr)']} 小時</strong>的中期玩家，建議在下個大改版前提出對策。</p>
              </div>
              <div class="action-card" style="border-top: 4px solid #4ecdc4; box-sizing: border-box;">
                <h4 style="color:#4ecdc4;">P2 第三優先：常態慢慢微調</h4>
                <p><strong>鎖定問題：{low_last['痛點屬性']}</strong>（急迫度 {low_last['痛點衝擊指數 (PPI)']} 分）</p>
                <p>抱怨這塊的玩家通常已玩超過 <strong>{low_last['負評中位數時數 (hr)']} 小時</strong>，屬於邊罵邊玩的核心鐵粉，排進平常例行更新調整即可，不用跟緊急 Bug 搶人力。</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            st.markdown("### 退坑玩家真實原話（紅底為系統自動抓出的痛點詞）")
            sel_asp = st.selectbox("切換想看的問題分類：", ppi_df["痛點屬性"].tolist())
            sub_rev = (
                df_absa[(df_absa["aspect"] == sel_asp) & (df_absa["is_churned_pain"])]
                .sort_values("votes_up", ascending=False).drop_duplicates(subset=["text"]).head(4)
            )
            if sub_rev.empty:
                sub_rev = (
                    df_absa[(df_absa["aspect"] == sel_asp) & (df_absa["is_neg"])]
                    .sort_values("votes_up", ascending=False).drop_duplicates(subset=["text"]).head(4)
                )
            for _, r_row in sub_rev.iterrows():
                hl_text = highlight_keywords(r_row["text"][:320], sel_asp)
                st.markdown(f"""
                <div class="review-card">
                    <div class="review-meta">{r_row['aspect']} ｜ 當時玩了：{r_row['playtime_hrs']} 小時 ｜ 階段：{r_row['lifecycle']} ｜ {r_row['votes_up']} 位玩家按讚認同</div>
                    <div>{hl_text}</div>
                </div>""", unsafe_allow_html=True)

        # --- TAB 2: 玩家什麼時候棄坑？ ---
        with tabs[1]:
            st.markdown("""
            <div class="explain-strip">
                <strong>為什麼要看遊玩時數？</strong> Steam 規定「購買 14 天內且遊玩未滿 2 小時」可無條件退款。<br>
                透過這兩張圖，遊戲公司可以一眼看出：哪些問題會害玩家<strong>剛買就退款（紅色）</strong>，哪些問題則是<strong>讓老玩家慢慢心寒（綠色）</strong>。
            </div>
            """, unsafe_allow_html=True)

            churned_absa = df_absa[df_absa["is_churned_pain"]].copy()
            e1, e2 = st.columns(2)
            with e1:
                if not churned_absa.empty:
                    lc_group = churned_absa.groupby(["aspect", "lifecycle"]).size().reset_index(name="退坑玩家人數")
                    lc_group = lc_group.rename(columns={"aspect": "問題分類", "lifecycle": "退坑階段"})
                    fig_lc = px.bar(
                        lc_group, x="問題分類", y="退坑玩家人數", color="退坑階段", barmode="stack",
                        color_discrete_map={
                            "1. 剛買就退坑 (≤2小時退款期)": "#ff4b4b",
                            "2. 玩沒幾天棄坑 (2~20小時)": "#ffa15a",
                            "3. 老玩家心寒 (>20小時)": "#4ecdc4"
                        },
                        template="plotly_dark", title="每個問題分別害「哪個時期的玩家」退坑"
                    )
                    fig_lc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)", height=370)
                    st.plotly_chart(fig_lc, use_container_width=True)

            with e2:
                neg_absa_eda = df_absa[df_absa["is_neg"]].copy()
                if not neg_absa_eda.empty:
                    neg_absa_eda = neg_absa_eda.rename(columns={"aspect": "問題分類", "playtime_hrs": "寫負評時玩了幾小時"})
                    fig_box = px.box(
                        neg_absa_eda, x="寫負評時玩了幾小時", y="問題分類", color="問題分類",
                        log_x=True, template="plotly_dark",
                        title="負評玩家都玩了多久才受不了（紅虛線＝2 小時退款門檻）"
                    )
                    fig_box.add_vline(x=2.0, line_dash="dash", line_color="#ff4b4b",
                                      annotation_text="Steam 2小時退款紅線", annotation_position="top right")
                    fig_box.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)",
                                          yaxis_title="", height=370)
                    st.plotly_chart(fig_box, use_container_width=True)

            daily_df = df_reviews.groupby("date").agg(
                每日負評總數=("voted_up", lambda s: (~s).sum()),
                每日真的退坑人數=("is_churned", "sum")
            ).reset_index().sort_values("date")

            if len(daily_df) > 1:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=daily_df["date"].astype(str), y=daily_df["每日負評總數"], mode="lines+markers", name="當天負評總數", line=dict(color="#ffa15a", width=3)))
                fig_ts.add_trace(go.Scatter(x=daily_df["date"].astype(str), y=daily_df["每日真的退坑人數"], mode="lines+markers", name="罵完真的退坑人數", line=dict(color="#ff4b4b", width=3)))
                fig_ts.update_layout(
                    template="plotly_dark", title="每天的負評數 vs. 真的退坑人數變化趨勢（可觀察哪天改版出包）",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)", hovermode="x unified", height=330
                )
                st.plotly_chart(fig_ts, use_container_width=True)

        # --- TAB 3: 同類型遊戲大 PK ---
        with tabs[2]:
            st.markdown("""
            <div class="explain-strip">
                <strong>同類型競品對比：</strong> 將兩款同類型遊戲的五大弱點疊在同一張雷達圖上；越靠外圈的地方代表那方面的退坑災情越嚴重，能直接看出競爭對手的強項與弱項。
            </div>
            """, unsafe_allow_html=True)

            if st.session_state["pk_enabled"] and st.session_state["pk_df"] is not None:
                df_pk = st.session_state["pk_df"].copy()
                df_pk["is_churned"] = (~df_pk["voted_up"].astype(bool)) & (df_pk["post_review_mins"] <= churn_mins)
                pk_name = st.session_state["pk_name"]
                pk_appid = st.session_state["pk_appid"]

                absa_pk, _ = run_sentence_absa(df_pk)
                ppi_pk = calculate_ppi(df_pk, absa_pk, prior_strength=float(prior_c))

                p_img1, p_img2 = st.columns(2)
                with p_img1:
                    h1 = get_game_header_url(selected_appid)
                    if h1:
                        st.image(h1, caption=f"[藍線] 主體檢遊戲：{selected_name}", use_container_width=True)
                    else:
                        st.info(f"[藍線] {selected_name}")
                with p_img2:
                    h2 = get_game_header_url(pk_appid)
                    if h2:
                        st.image(h2, caption=f"[紅線] 對照組競品：{pk_name}", use_container_width=True)
                    else:
                        st.info(f"[紅線] {pk_name}")

                all_asps = list(ASPECT_KEYWORDS.keys())
                s1_map = dict(zip(ppi_df["痛點屬性"], ppi_df["痛點衝擊指數 (PPI)"]))
                s2_map = dict(zip(ppi_pk["痛點屬性"], ppi_pk["痛點衝擊指數 (PPI)"]))
                v1 = [s1_map.get(a, 0) for a in all_asps]
                v2 = [s2_map.get(a, 0) for a in all_asps]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=v1 + [v1[0]], theta=all_asps + [all_asps[0]], fill="toself", name=selected_name, line_color="#66c0f4"))
                fig_radar.add_trace(go.Scatterpolar(r=v2 + [v2[0]], theta=all_asps + [all_asps[0]], fill="toself", name=pk_name, line_color="#ff4b4b"))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    template="plotly_dark", title="同類型競品五大痛點雷達圖對決（越靠外圈越嚴重）",
                    paper_bgcolor="rgba(0,0,0,0)", height=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("目前為單款遊戲體檢模式。請點擊上方「重選／更換比較遊戲」或在左側開啟「開啟雙遊戲 PK 比較」選擇同類型對手即可對決。")

        # --- TAB 4: 資料科學與統計驗證 ---
        with tabs[3]:
            st.markdown("""
            <div class="explain-strip">
                <strong>資料科學核心方法論驗證專區：</strong> 完整呈現背後的統計推論（勝算比 95% CI 森林圖、卡方檢定、貝葉斯平滑）、監督式機器學習模型評估（ROC-AUC、XAI 係數）以及參數敏感度穩健性檢驗（Kendall's W）。
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 1. 統計推論：勝算比 95% 信賴區間森林圖 & 貝葉斯小樣本平滑")
            s1, s2 = st.columns(2)
            with s1:
                f_df = ppi_df.sort_values("_or_num", ascending=True).copy()
                fig_forest = go.Figure()
                fig_forest.add_trace(go.Scatter(
                    x=f_df["_or_num"], y=f_df["痛點屬性"], mode="markers",
                    marker=dict(size=12, color=["#ff4b4b" if v > 1.0 else "#4ecdc4" for v in f_df["_or_num"]]),
                    error_x=dict(
                        type="data", symmetric=False,
                        array=(f_df["_ci_high"] - f_df["_or_num"]).clip(lower=0),
                        arrayminus=(f_df["_or_num"] - f_df["_ci_low"]).clip(lower=0),
                        color="#c7d5e0", thickness=2
                    ),
                    hovertemplate="<b>%{y}</b><br>Odds Ratio: %{x:.2f}x<extra></extra>"
                ))
                fig_forest.add_vline(x=1.0, line_dash="dash", line_color="#ffa15a",
                                     annotation_text="OR = 1.0 (無顯著影響線)", annotation_position="top right")
                fig_forest.update_layout(
                    xaxis_type="log", template="plotly_dark",
                    title="流失勝算比 (Odds Ratio) 95% CI 森林圖（對數軸）",
                    xaxis_title="Odds Ratio (>1 代表顯著增加退坑風險)", yaxis_title="",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)", height=340
                )
                st.plotly_chart(fig_forest, use_container_width=True)

            with s2:
                melt_b = ppi_df.melt(id_vars="痛點屬性", value_vars=["原始負評率 (%)", "平滑負評率 (%)"], var_name="估計方法", value_name="負評率 (%)")
                fig_bayes = px.bar(melt_b, x="痛點屬性", y="負評率 (%)", color="估計方法", barmode="group",
                                   color_discrete_map={"原始負評率 (%)": "#6c7a89", "平滑負評率 (%)": "#66c0f4"},
                                   template="plotly_dark", title=f"貝葉斯平滑校正前後對照 (先驗強度 C = {prior_c})")
                fig_bayes.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)",
                                        xaxis_title="", height=340)
                st.plotly_chart(fig_bayes, use_container_width=True)

            show_cols = ["痛點屬性", "提及次數", "負面提及數", "平滑負評率 (%)", "平滑流失轉化率 (%)", "流失勝算比 (OR)", "OR 95% 信賴區間", "卡方檢定 p-value", "負評中位數時數 (hr)", "痛點衝擊指數 (PPI)"]
            st.dataframe(ppi_df[show_cols], use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 2. 監督式機器學習模型評估（ROC-AUC）與特徵可解釋性（XAI）")
            if ml_ready:
                lr_prob = lr_pipe.predict_proba(X_te)[:, 1]
                rf_prob = rf_clf.predict_proba(X_te)[:, 1]
                rf_pred = rf_clf.predict(X_te)

                lr_auc = roc_auc_score(y_te, lr_prob)
                rf_auc = roc_auc_score(y_te, rf_prob)
                rf_acc = accuracy_score(y_te, rf_pred)
                rf_f1 = f1_score(y_te, rf_pred, zero_division=0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Random Forest AUC", f"{rf_auc:.3f}")
                m2.metric("Logistic Regression AUC", f"{lr_auc:.3f}")
                m3.metric("測試集準確率 (Accuracy)", f"{rf_acc*100:.1f}%")
                m4.metric("流失預測 F1-Score", f"{rf_f1:.3f}")

                ml_c1, ml_c2 = st.columns([1.0, 1.2])
                with ml_c1:
                    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_prob)
                    fpr_rf, tpr_rf, _ = roc_curve(y_te, rf_prob)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines", name=f"Random Forest (AUC={rf_auc:.3f})", line=dict(color="#66c0f4", width=3)))
                    fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines", name=f"Logistic Reg (AUC={lr_auc:.3f})", line=dict(color="#4ecdc4", width=2.5)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="隨機基準線 (0.50)", line=dict(color="#8f98a0", dash="dash")))
                    fig_roc.update_layout(template="plotly_dark", title="流失預測模型 ROC-AUC 曲線",
                                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)", height=340)
                    st.plotly_chart(fig_roc, use_container_width=True)

                with ml_c2:
                    coefs = lr_pipe.named_steps["clf"].coef_[0]
                    imp_df = pd.DataFrame({
                        "特徵變數": X_df.columns,
                        "標準化邏輯迴歸係數": np.round(coefs, 3),
                        "影響方向": ["正向推升退坑風險" if c > 0 else "保護留存／降低流失" for c in coefs]
                    }).sort_values("標準化邏輯迴歸係數", ascending=True)
                    fig_imp = px.bar(
                        imp_df, x="標準化邏輯迴歸係數", y="特徵變數", orientation="h",
                        color="影響方向", color_discrete_map={"正向推升退坑風險": "#ff4b4b", "保護留存／降低流失": "#4ecdc4"},
                        template="plotly_dark", title="模型可解釋性 (XAI)：各痛點對退坑的真實邊際影響"
                    )
                    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)",
                                          yaxis_title="", height=340)
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("目前這組資料類別分布過於偏斜，請調整參數或更換遊戲檢視模型評估結果。")

            st.divider()
            st.markdown("#### 3. 痛點共現關聯與參數敏感度驗證（Kendall's W 排序一致性）")
            v1, v2 = st.columns([1.0, 1.2])
            with v1:
                neg_only_co = df_absa[df_absa["is_neg"]]
                if not neg_only_co.empty:
                    piv_co = pd.crosstab(neg_only_co["review_idx"], neg_only_co["aspect"]).clip(upper=1)
                    co_mat = piv_co.T.dot(piv_co)
                    fig_heat = px.imshow(
                        co_mat, text_auto=True, color_continuous_scale="Blues",
                        template="plotly_dark", title="負面痛點共現矩陣 (Co-occurrence)"
                    )
                    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=340)
                    st.plotly_chart(fig_heat, use_container_width=True)

            with v2:
                sens_df, kendall_w = run_sensitivity_kendall_w(df_reviews, prior_strength=float(prior_c))
                if not sens_df.empty:
                    fig_sens = px.line(
                        sens_df, x="門檻 ΔT (分鐘)", y="PPI 分數", color="痛點屬性", markers=True,
                        template="plotly_dark",
                        title=f"敏感度檢定：不同 ΔT 門檻下排名穩健性 (Kendall's W = {kendall_w})"
                    )
                    fig_sens.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23,26,33,0.6)", height=340)
                    st.plotly_chart(fig_sens, use_container_width=True)
