import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from datetime import datetime, timedelta

# -------------------- 数据加载 --------------------
@st.cache_data
def load_data(csv_path="p3_history.csv"):
    df = pd.read_csv(csv_path, names=['draw_date','draw_num','n1','n2','n3'], header=None)
    df['draw_date'] = pd.to_datetime(df['draw_date'])
    df.sort_values('draw_date', inplace=True, ignore_index=True)
    df['form'] = df.apply(lambda r: '豹子' if r.n1==r.n2==r.n3 else ('组三' if r.n1==r.n2 or r.n1==r.n3 or r.n2==r.n3 else '组六'), axis=1)
    df['sum_val'] = df[['n1','n2','n3']].sum(axis=1)
    df['span'] = df[['n1','n2','n3']].max(axis=1) - df[['n1','n2','n3']].min(axis=1)
    df['odd_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v%2 for v in x), axis=1)
    df['big_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v>=5 for v in x), axis=1)
    df['r0_cnt'] = df[['n1','n2','n3']].apply(lambda x: sum(v%3==0 for v in x), axis=1)
    return df

# -------------------- 评分卡 --------------------
def compute_scorecard(df, back_period=30):
    recent = df.tail(back_period)
    scores = {}
    for digit in range(10):
        appearances = ((recent['n1']==digit) | (recent['n2']==digit) | (recent['n3']==digit)).sum()
        last_idx = -1
        for i, row in recent[::-1].iterrows():
            if digit in (row.n1, row.n2, row.n3):
                last_idx = len(recent) - 1 - (list(recent.index).index(i))
                break
        skip = len(recent) - last_idx - 1 if last_idx != -1 else len(recent)
        inherit = 0
        for i in range(1, len(recent)):
            prev = recent.iloc[i-1]
            curr = recent.iloc[i]
            if digit in (prev.n1,prev.n2,prev.n3) and digit in (curr.n1,curr.n2,curr.n3):
                inherit += 1
        small_bonus = 1 if digit < 5 else 0
        big_bonus = 1 if digit >= 5 else 0
        odd_bonus = 1 if digit%2 else 0
        even_bonus = 1 if digit%2==0 else 0
        freq_score = appearances/back_period*10
        skip_score = 1/(skip+1)*10
        inherit_score = inherit/(back_period-1)*10
        balance_score = (small_bonus+big_bonus+odd_bonus+even_bonus)/4*5
        total = 0.3*freq_score + 0.3*skip_score + 0.2*inherit_score + 0.2*balance_score
        scores[digit] = round(total,2)
    score_df = pd.DataFrame({'数字':list(scores.keys()),'评分':list(scores.values())})
    return score_df.sort_values('评分',ascending=False)

def generate_pick5(score_df):
    top5 = score_df.head(5)['数字'].tolist()
    def imbalance(digits):
        odds = sum(1 for d in digits if d%2)
        bigs = sum(1 for d in digits if d>=5)
        return odds>4 or odds<1 or bigs>4 or bigs<1
    final = top5.copy()
    if imbalance(final):
        for d in score_df.head(7)['数字']:
            if d not in final:
                final[-1]=d
                if not imbalance(final): break
    return sorted(final)

# -------------------- AI 训练与预测 --------------------
def make_features(df, start_idx, end_idx):
    """为[start_idx, end_idx)的每一行构建特征，返回特征矩阵和下一期多标签向量"""
    X, y = [], []
    for i in range(start_idx, end_idx):
        if i < 5: continue  # 需要至少5期历史才能构建特征
        # 取当前行之前的窗口数据（不含当前行）
        past = df.iloc[max(0,i-30):i]
        feats = {}
        # 近期频次
        for d in range(10):
            cnt = ((past['n1']==d)|(past['n2']==d)|(past['n3']==d)).sum()
            feats[f'freq_{d}'] = cnt / len(past) if len(past)>0 else 0
        # 遗漏
        for d in range(10):
            last = -1
            for j in range(len(past)-1,-1,-1):
                if d in (past.iloc[j].n1, past.iloc[j].n2, past.iloc[j].n3):
                    last = len(past)-1-j
                    break
            skip = len(past)-last-1 if last!=-1 else len(past)
            feats[f'skip_{d}'] = skip
        # 上期直落
        if i>0:
            prev = df.iloc[i-1]
            for d in range(10):
                feats[f'prev_{d}'] = 1 if d in (prev.n1,prev.n2,prev.n3) else 0
        # 基本形态统计
        feats['sum_mean'] = past['sum_val'].mean() if len(past)>0 else 13.5
        feats['span_mean'] = past['span'].mean() if len(past)>0 else 4.5
        feats['odd_ratio'] = past['odd_cnt'].mean()/3 if len(past)>0 else 0.5
        feats['big_ratio'] = past['big_cnt'].mean()/3 if len(past)>0 else 0.5
        # 日期特征
        date = df.iloc[i]['draw_date']
        feats['dayofweek'] = date.weekday()
        feats['dayofyear'] = date.timetuple().tm_yday

        X.append(feats)
        # 标签：当前行的多标签向量
        curr = df.iloc[i]
        label = [1 if d in (curr.n1,curr.n2,curr.n3) else 0 for d in range(10)]
        y.append(label)
    return pd.DataFrame(X), np.array(y)

@st.cache_resource
def train_ai_model(df):
    """训练XGB多标签模型，返回模型和训练信息"""
    if len(df) < 20:
        return None, "数据不足（至少20期）", None
    X, y = make_features(df, 0, len(df))
    if len(X) < 10:
        return None, "可训练样本不足", None

    # 时序交叉验证评估
    tscv = TimeSeriesSplit(n_splits=3)
    model = MultiOutputClassifier(XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0, random_state=42))
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        # 用子样本准确率（每个数字的准确率均值）
        pred = model.predict(X_test)
        acc = np.mean([np.mean(pred[:,d] == y_test[:,d]) for d in range(10)])
        scores.append(acc)
    avg_acc = np.mean(scores) if scores else 0
    # 最后用全量数据重新训练
    model.fit(X, y)
    return model, f"交叉验证平均精确度: {avg_acc:.2%}", X.columns.tolist()

def predict_next(model, df):
    """预测最新一期的下一期概率"""
    if model is None or len(df) < 5:
        return None
    # 构建最新一期的特征（用整个历史数据构造，取最后一行）
    X_latest, _ = make_features(df, len(df)-1, len(df))  # 只预测最新一期的下一期
    if X_latest.empty:
        return None
    prob = model.predict_proba(X_latest.iloc[[0]])  # 返回列表，每个元素是 [neg_proba, pos_proba]
    prob_dict = {d: prob[d][0][1] for d in range(10)}  # 取正类概率
    return prob_dict

# -------------------- UI --------------------
st.set_page_config(page_title="组选五码决策系统", layout="wide")
st.title("🎯 排列三 · 组选五码智能决策仪表盘")
st.caption("量化评分 + AI概率融合，每日优选五码")

df = load_data()
if df.empty:
    st.error("数据加载失败，请检查 p3_history.csv")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 控制面板")
back_days = st.sidebar.slider("评分卡回溯期数", 15, 100, 30, 5)
ai_weight = st.sidebar.slider("AI引擎权重", 0.0, 1.0, 0.4, 0.05)
st.sidebar.markdown(f"---\n历史数据总量：**{len(df)}** 期")
latest_date = df['draw_date'].iloc[-1].strftime('%Y-%m-%d')
st.sidebar.markdown(f"最新一期：{latest_date}")

# 训练AI模型
with st.spinner("正在训练AI预测模型..."):
    ai_model, ai_msg, feature_cols = train_ai_model(df)
    if ai_model:
        st.sidebar.success(f"AI模型就绪 | {ai_msg}")
        # 预测下一期概率
        ai_probs = predict_next(ai_model, df)
    else:
        st.sidebar.warning("AI模型未训练：" + ai_msg)
        ai_probs = None

# 评分卡
score_df = compute_scorecard(df, back_days)
quant_top5 = generate_pick5(score_df)

# 融合推荐
if ai_probs:
    # 将评分卡和AI概率加权融合
    score_df['AI概率'] = score_df['数字'].apply(lambda d: ai_probs.get(d, 0))
    # 归一化评分和概率到0-1区间，再融合
    max_s = score_df['评分'].max()
    max_p = score_df['AI概率'].max()
    score_df['评分_norm'] = score_df['评分'] / max_s if max_s>0 else 0
    score_df['AI_norm'] = score_df['AI概率'] / max_p if max_p>0 else 0
    score_df['融合分'] = (1-ai_weight)*score_df['评分_norm'] + ai_weight*score_df['AI_norm']
    score_df.sort_values('融合分', ascending=False, inplace=True)
    fusion_top5 = score_df.head(5)['数字'].tolist()
    fusion_top5 = sorted(fusion_top5)
else:
    fusion_top5 = quant_top5

# 主界面
col1, col2 = st.columns([1, 1.5])
with col1:
    st.subheader("📊 数字评分卡")
    st.bar_chart(score_df.set_index('数字')['评分'], use_container_width=True)
    st.caption("基于冷热、遗漏、遗传的量化评分")
with col2:
    st.subheader("🏆 最终五码推荐")
    st.markdown(f"### {', '.join(map(str, fusion_top5))}")
    st.caption("（量化+AI 加权融合）")
    if ai_probs:
        st.markdown("---")
        st.metric("AI最高概率数字", max(ai_probs, key=ai_probs.get), f"{ai_probs[max(ai_probs, key=ai_probs.get)]:.1%}")
    st.metric("纯量化最高分数字", score_df.sort_values('评分',ascending=False).iloc[0]['数字'])

# 详细数据
tab1, tab2 = st.tabs(["📋 详细评分表", "🤖 AI概率详情"])
with tab1:
    st.dataframe(score_df.style.format({'评分':'{:.1f}','AI概率':'{:.1%}','融合分':'{:.3f}'}), use_container_width=True, hide_index=True)
with tab2:
    if ai_probs:
        prob_df = pd.DataFrame({'数字':list(range(10)), '概率':list(ai_probs.values())})
        prob_df.sort_values('概率', ascending=False, inplace=True)
        st.bar_chart(prob_df.set_index('数字')['概率'], use_container_width=True)
        st.dataframe(prob_df.style.format({'概率':'{:.2%}'}), hide_index=True)
    else:
        st.info("AI模型未就绪，无法显示概率详情。")

st.markdown("---")
st.caption("🔮 下一步将融合玄学能量权重与交易仓位管理。当前版本：引擎一（量化）+ 引擎二（AI）")
