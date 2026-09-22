"""结直肠癌术后 DVT 风险预测系统 / CRC Postoperative DVT Risk Predictor
模型：随机森林（Random Forest）｜ LASSO lambda.1se 筛选的 5 项围术期指标
训练集 n=455 ｜ 验证集 n=195 ｜ 独立验证集 n=187
支持中英文双语切换 / Bilingual (Chinese / English)
"""
import streamlit as st
import streamlit.components.v1 as components
import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
st.set_page_config(page_title="CRC术后DVT风险预测 / DVT Risk Predictor",
                   page_icon="🏥", layout="wide")


def st_shap(plot, height=170):
    components.html(f"<head>{shap.getjs()}</head><body>{plot.html()}</body>",
                    height=height, scrolling=True)


@st.cache_resource
def load():
    base = 'saved_models'
    model = joblib.load(f'{base}/model.pkl')
    scaler = joblib.load(f'{base}/scaler.pkl')
    feats = list(joblib.load(f'{base}/feature_names.pkl'))
    return model, scaler, feats


model, scaler, FEATURES = load()

TEXT = {
    'zh': {
        'title': "🏥 结直肠癌术后 DVT 风险预测系统",
        'caption': "随机森林 ｜ LASSO (lambda.1se) 筛选的 5 项围术期指标 ｜ "
                   "验证集 AUC 0.746 ｜ 独立验证集 AUC 0.790 ｜ "
                   "训练 n=455 · 验证 n=195 · 独立验证 n=187",
        'tab1': "📝 单例预测", 'tab2': "📂 批量预测",
        'info1': "输入 5 项指标，评估结直肠癌根治术后 14 天内下肢深静脉血栓（DVT）风险",
        'btn': "🚀 评估 DVT 风险",
        'metric': "DVT 风险概率",
        'low': "🟢 低风险", 'mid': "🟡 中风险", 'high': "🔴 高风险",
        'low_msg': "常规围术期预防即可，继续观察",
        'mid_msg': "建议术后加强下肢超声监测",
        'high_msg': "⚠️ 建议加强超声监测，并评估血栓预防方案",
        'shap_title': "🔍 SHAP 归因分析",
        'calc': "计算特征贡献...",
        'force': "**互动力图**（鼠标悬停查看细节）",
        'expand': "查看输入与标准化后数值",
        'raw': "原始值", 'std': "标准化",
        'info2': "上传包含以下特征列的 Excel / CSV 文件，批量评估 DVT 风险",
        'tpl': "📥 下载模板", 'upload': "上传文件",
        'missing': "缺少列", 'risk': "DVT风险", 'level': "风险等级",
        'hl': "🔴高风险", 'ml': "🟡中风险", 'll': "🟢低风险",
        'dl': "💾 下载结果", 'single': "🔍 单样本解释",
        'pick': "选择行号", 'exp_btn': "解释此样本",
        'foot': "模型参数：随机森林，Optuna 50 轮贝叶斯优化，未使用过采样，阈值基于 Youden 指数。"
                "本工具仅供临床研究参考，不作为独立诊疗依据。",
        'labels': {'D_dimer': 'D-二聚体（mg/L）', 'Age': '年龄（岁）',
                   'SII': '全身免疫炎症指数 SII', 'NLR': '中性粒细胞/淋巴细胞比值 NLR',
                   'Fib': '纤维蛋白原（g/L）'},
        'help': {'D_dimer': '术前 D-二聚体，正常参考值 <0.5 mg/L', 'Age': '手术时年龄',
                 'SII': 'SII = 血小板 × 中性粒细胞 / 淋巴细胞',
                 'NLR': 'NLR = 中性粒细胞 / 淋巴细胞',
                 'Fib': '术前纤维蛋白原，正常参考值 2–4 g/L'},
    },
    'en': {
        'title': "🏥 CRC Postoperative DVT Risk Predictor",
        'caption': "Random Forest ｜ 5 perioperative variables selected by LASSO (lambda.1se) ｜ "
                   "Validation AUC 0.746 ｜ Independent AUC 0.790 ｜ "
                   "Training n=455 · Validation n=195 · Independent n=187",
        'tab1': "📝 Single-case prediction", 'tab2': "📂 Batch prediction",
        'info1': "Enter 5 variables to estimate the risk of lower-extremity deep vein thrombosis "
                 "(DVT) within 14 days after radical resection for colorectal cancer",
        'btn': "🚀 Assess DVT risk",
        'metric': "DVT risk probability",
        'low': "🟢 Low risk", 'mid': "🟡 Intermediate risk", 'high': "🔴 High risk",
        'low_msg': "Routine perioperative prophylaxis is sufficient; continue observation",
        'mid_msg': "Consider intensified postoperative lower-limb ultrasound surveillance",
        'high_msg': "⚠️ Intensified ultrasound surveillance and reassessment of "
                    "thromboprophylaxis are recommended",
        'shap_title': "🔍 SHAP attribution",
        'calc': "Computing feature contributions...",
        'force': "**Interactive force plot** (hover for details)",
        'expand': "View input and standardised values",
        'raw': "Raw value", 'std': "Standardised",
        'info2': "Upload an Excel/CSV file containing the following columns for batch assessment",
        'tpl': "📥 Download template", 'upload': "Upload file",
        'missing': "Missing columns", 'risk': "DVT risk", 'level': "Risk category",
        'hl': "🔴 High", 'ml': "🟡 Intermediate", 'll': "🟢 Low",
        'dl': "💾 Download results", 'single': "🔍 Single-sample explanation",
        'pick': "Select row", 'exp_btn': "Explain this sample",
        'foot': "Model: Random Forest, Optuna Bayesian optimisation (50 trials), "
                "no oversampling applied, cut-off based on the Youden index. "
                "This tool is intended for clinical research reference only and "
                "must not be used as a standalone basis for diagnosis or treatment.",
        'labels': {'D_dimer': 'D-dimer (mg/L)', 'Age': 'Age (years)',
                   'SII': 'Systemic immune-inflammation index (SII)',
                   'NLR': 'Neutrophil-to-lymphocyte ratio (NLR)',
                   'Fib': 'Fibrinogen (g/L)'},
        'help': {'D_dimer': 'Preoperative D-dimer; reference <0.5 mg/L', 'Age': 'Age at surgery',
                 'SII': 'SII = platelet × neutrophil / lymphocyte',
                 'NLR': 'NLR = neutrophil / lymphocyte',
                 'Fib': 'Preoperative fibrinogen; reference 2–4 g/L'},
    },
}
DEFAULTS = {'D_dimer': 0.85, 'Age': 64.0, 'SII': 728.0, 'NLR': 2.75, 'Fib': 3.24}
THR_HIGH, THR_MID = 0.250, 0.100

lang = st.sidebar.radio("语言 / Language", ["中文", "English"], index=0)
L = 'zh' if lang == "中文" else 'en'
T = TEXT[L]
LB, HP = T['labels'], T['help']


@st.cache_resource
def get_explainer():
    # TreeExplainer 默认的 tree_path_dependent 模式无需背景样本，
    # 因此不依赖任何训练数据文件（避免在仓库中存放患者数据）
    return shap.TreeExplainer(model)


explainer = get_explainer()

st.title(T['title'])
st.caption(T['caption'])

tab1, tab2 = st.tabs([T['tab1'], T['tab2']])

with tab1:
    st.info(T['info1'])
    with st.form("form"):
        inputs = {}
        cols = st.columns(3)
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                inputs[f] = st.number_input(LB.get(f, f), value=float(DEFAULTS.get(f, 0)),
                                            format="%.2f", help=HP.get(f, ''))
        ok = st.form_submit_button(T['btn'])

    if ok:
        x = pd.DataFrame([inputs], columns=FEATURES)
        xs = scaler.transform(x)
        prob = float(model.predict_proba(xs)[0, 1])

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(T['metric'], f"{prob*100:.1f}%")
            if prob < THR_MID:
                st.success(T['low']); st.info(T['low_msg'])
            elif prob < THR_HIGH:
                st.warning(T['mid']); st.info(T['mid_msg'])
            else:
                st.error(T['high']); st.warning(T['high_msg'])

        with c2:
            st.subheader(T['shap_title'])
            with st.spinner(T['calc']):
                try:
                    sv = explainer(pd.DataFrame(xs, columns=FEATURES), check_additivity=False)
                    vals = sv.values[0]
                    if np.ndim(vals) > 1: vals = vals[:, 1]
                    bv = sv.base_values[0]
                    if np.ndim(bv) > 0: bv = float(np.ravel(bv)[-1])
                    exp = shap.Explanation(values=vals, base_values=float(bv),
                                           data=x.iloc[0].values, feature_names=FEATURES)
                    fig = plt.figure(figsize=(10, 3.8))
                    shap.plots.waterfall(exp, max_display=len(FEATURES), show=False)
                    plt.title('SHAP Waterfall', fontweight='bold')
                    st.pyplot(fig, bbox_inches='tight'); plt.close(fig)
                    st.markdown(T['force'])
                    fh = shap.plots.force(float(bv), np.round(vals, 3),
                                          np.round(x.iloc[0].values, 3),
                                          feature_names=FEATURES, matplotlib=False)
                    st_shap(fh)
                except Exception as e:
                    st.warning(f"SHAP error: {e}")

        with st.expander(T['expand']):
            st.write(pd.DataFrame({T['raw']: x.iloc[0].values,
                                   T['std']: np.round(xs[0], 3)}, index=FEATURES))

with tab2:
    st.info(T['info2'])
    st.code(str(FEATURES))
    st.download_button(T['tpl'],
                       pd.DataFrame(columns=FEATURES).to_csv(index=False).encode('utf-8-sig'),
                       "template.csv")
    uf = st.file_uploader(T['upload'], type=["xlsx", "csv"])
    if uf:
        df_u = pd.read_excel(uf) if uf.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uf)
        missing = [c for c in FEATURES if c not in df_u.columns]
        if missing:
            st.error(f"{T['missing']}: {missing}")
        else:
            xs_b = scaler.transform(df_u[FEATURES].astype(float))
            probs = model.predict_proba(xs_b)[:, 1]
            df_u[T['risk']] = probs.round(4)
            df_u[T['level']] = [T['hl'] if p >= THR_HIGH else T['ml'] if p >= THR_MID
                                else T['ll'] for p in probs]
            st.dataframe(df_u, use_container_width=True)
            st.download_button(T['dl'],
                               df_u.to_csv(index=False).encode('utf-8-sig'), "result.csv")
            st.divider()
            st.subheader(T['single'])
            idx = st.selectbox(T['pick'], df_u.index)
            if st.button(T['exp_btn']):
                x_s = df_u.loc[[idx], FEATURES].astype(float)
                xs_s = scaler.transform(x_s)
                sv_s = explainer(pd.DataFrame(xs_s, columns=FEATURES), check_additivity=False)
                v_s = sv_s.values[0]
                if np.ndim(v_s) > 1: v_s = v_s[:, 1]
                b_s = sv_s.base_values[0]
                if np.ndim(b_s) > 0: b_s = float(np.ravel(b_s)[-1])
                exp_s = shap.Explanation(values=v_s, base_values=float(b_s),
                                         data=x_s.iloc[0].values, feature_names=FEATURES)
                fig = plt.figure(figsize=(10, 3.8))
                shap.plots.waterfall(exp_s, max_display=len(FEATURES), show=False)
                st.pyplot(fig, bbox_inches='tight'); plt.close(fig)

st.divider()
st.caption(T['foot'])
