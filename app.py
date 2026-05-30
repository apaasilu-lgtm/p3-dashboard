import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime

# -------------------- 数据加载（从CSV动态建库） --------------------
@st.cache_data
def load_data(csv_path="p3_history.csv"):
    """读取CSV历史数据，并计算所需特征，返回完整DataFrame"""
    try:
        df = pd.read_csv(csv_path, names=['draw_date','draw_num','n1','n2','n3'], header=None)
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df.sort_values('draw_date', inplace=True, ignore_index=True)

        # 基础特征（与之前一致）
        df['form'] = df.apply(lambda r: '豹子' if r.n1==r.n2==r.n3 else ('组三' if r.n1==r.n2 or r.n1==r.n3 or r.n2==r.n3 else '组六'), axis=1)
        df['sum_val'] = df[['n1','n2','n3']].sum(axis=1)
        df['span'] = df[['n1','n2','n3']].max(axis=1) - df[['n1','n2','n3']].min(axis=1)
        df['odd_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v%2 for v in x), axis=1)
        df['big_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v>=5 for v in x), axis=1)
        df['r0_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v%3==0 for v in x), axis=1)
        return df
    except Exception as e:
        st.error(f"数据加载失败：{e}，请检查 p3_history.csv 格式。")
        return pd.DataFrame()

# -------------------- 评分卡核心函数 --------------------
def compute_scorecard(df, back_period=30):
    """为0-9每个数字计算状态评分，返回评分DataFrame"""
    if df.empty:
        return pd.DataFrame()
    recent = df.tail(back_period)
    scores = {}
    for digit in range(10):
        # 近期出现频次
        appearances = ((recent['n1']==digit) | (recent['n2']==digit) | (recent['n3']==digit)).sum()
        # 遗漏期数
        last_idx = -1
        for i, row in recent[::-1].iterrows():
            if digit in (row.n1, row.n2, row.n3):
                last_idx = len(recent) - 1 - (list(recent.index).index(i))
                break
        skip = len(recent) - last_idx - 1 if last_idx != -1 else len(recent)

        # 近10期遗传次数（与上一期有相同数字）
        same_as_prev = 0
        for i in range(1, len(recent)):
            prev = recent.iloc[i-1]
            curr = recent.iloc[i]
            if digit in (prev.n1, prev.n2, prev.n3) and digit in (curr.n1, curr.n2, curr.n3):
                same_as_prev += 1

        # 均衡加分：0-4小，5-9大
        small_bonus = 1 if digit < 5 else 0
        big_bonus = 1 if digit >= 5 else 0
        # 奇偶加分
        odd_bonus = 1 if digit%2 else 0
        even_bonus = 1 if digit%2==0 else 0

        # 综合评分：频次 + 近期遗漏倒数 + 遗传趋势 + 均衡加权
        freq_score = appearances / back_period * 10
        skip_score = 1.0 / (skip+1) * 10  # 遗漏越短越好，但冷号长期遗漏此分会极低
        inherit_score = same_as_prev / (back_period-1) * 10
        balance_score = (small_bonus + big_bonus + odd_bonus + even_bonus) / 4 * 5

        total = 0.3*freq_score + 0.3*skip_score + 0.2*inherit_score + 0.2*balance_score
        scores[digit] = round(total, 2)

    score_df = pd.DataFrame({'数字':list(scores.keys()), '评分':list(scores.values())})
    return score_df.sort_values('评分', ascending=False)

# -------------------- 五码生成建议 --------------------
def generate_pick5(score_df, balance_rule=True):
    """根据评分生成一组组选五码，并考虑均衡性调整"""
    top5 = score_df.head(5)['数字'].tolist()
    if not balance_rule:
        return sorted(top5)
    # 轻度均衡：如果奇偶或大小严重失衡，从6-7名替换最末尾
    def imbalance_check(digits):
        odds = sum(1 for d in digits if d%2)
        bigs = sum(1 for d in digits if d>=5)
        return odds>4 or odds<1 or bigs>4 or bigs<1
    final = top5.copy()
    if imbalance_check(final):
        for d in score_df.head(7)['数字'].tolist():
            if d not in final:
                final[-1] = d
                if not imbalance_check(final):
                    break
    return sorted(final)

# -------------------- UI 界面 --------------------
st.set_page_config(page_title="组选五码决策仪表盘", layout="wide")
st.title("🎯 排列三 · 组选五码智能评分卡")
st.caption("基于冷热、遗漏、遗传、均衡的量化动态评分")

# 加载数据
df = load_data()
if df.empty:
    st.stop()

# 侧边栏控制
st.sidebar.header("⚙️ 分析参数")
back_days = st.sidebar.slider("回溯期数", min_value=15, max_value=100, value=30, step=5)
st.sidebar.markdown(f"---\n当前数据总量：**{len(df)}** 期\n最新一期：{df['draw_date'].iloc[-1].strftime('%Y-%m-%d')}")

# 计算评分卡
score_tab = compute_scorecard(df, back_days)
top5 = generate_pick5(score_tab)

# 主区域展示
col1, col2 = st.columns([1, 1.2])
with col1:
    st.subheader("📊 数字评分卡")
    st.bar_chart(score_tab.set_index('数字')['评分'], use_container_width=True)
with col2:
    st.subheader("🏆 今日五码推荐")
    st.markdown(f"### {','.join(map(str, top5))}")
    st.caption("（已自动均衡奇偶/大小）")
    st.markdown("---")
    st.metric("最高评分数字", score_tab.iloc[0]['数字'], f"{score_tab.iloc[0]['评分']}分")
    st.metric("次高评分数字", score_tab.iloc[1]['数字'], f"{score_tab.iloc[1]['评分']}分")

# 详细评分表格
st.subheader("📋 各数字详细评分明细")
st.dataframe(score_tab.style.format({'评分':'{:.1f}'}), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("🔮 下一阶段将融合AI预测、玄学权重与交易心理仓位，此仪表盘为引擎一·量化层核心。")
