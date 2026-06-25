"""CRC术后DVT预测系统 — Streamlit版 (10变量LightGBM)"""
import streamlit as st
import streamlit.components.v1 as components
import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt

def st_shap(plot, height=150):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

@st.cache_resource
def load():
    base = 'saved_models'
    model = joblib.load(f'{base}/LightGBMOpt.pkl')
    scaler = joblib.load(f'{base}/scaler.pkl')
    features = list(joblib.load(f'{base}/feature_names.pkl'))
    return model, scaler, features

model, scaler, features = load()

LABELS = {
    'AFR':'白蛋白/纤维蛋白原比 AFR','PNI':'预后营养指数 PNI',
    'Blood_transfusion':'围术期输血','Tumor_site':'肿瘤部位',
    'Varicose_veins':'静脉曲张','TT':'凝血酶时间 TT(秒)',
    'Age':'年龄(岁)','Operation_time':'手术时间(min)',
    'VTE_family_history':'VTE家族史','D_dimer':'D-二聚体(mg/L)'
}
DEFAULTS = {'AFR':13.5,'PNI':50.0,'Blood_transfusion':0,'Tumor_site':1,
            'Varicose_veins':0,'TT':13.5,'Age':62,'Operation_time':140,
            'VTE_family_history':0,'D_dimer':3.5}
CATS = ['Blood_transfusion','Tumor_site','Varicose_veins','VTE_family_history']
CAT_OPTS = {
    'Blood_transfusion':[0,1],'Tumor_site':[1,2],
    'Varicose_veins':[0,1],'VTE_family_history':[0,1]
}

st.set_page_config(page_title="CRC术后DVT预测",page_icon="🏥",layout="wide")
st.title("🏥 结直肠癌术后DVT风险预测系统")
st.caption(f"LightGBM | LASSO 10变量 | Val AUC 0.866 | Ext AUC 0.835")

# ---- Explainer ----
@st.cache_resource
def get_explainer():
    tr = pd.read_excel('train_data/synth_KDE_train_455.xlsx')
    fn = list(joblib.load('saved_models/feature_names.pkl'))
    VARS = ['AFR','PNI','Blood_transfusion','Tumor_site','Varicose_veins','TT','Age','Operation_time','VTE_family_history','D_dimer']
    Xtr = pd.get_dummies(tr[VARS], drop_first=True).reindex(columns=fn, fill_value=0)
    Xtr_s = scaler.transform(Xtr.values)
    bg = shap.sample(pd.DataFrame(Xtr_s, columns=fn), 100, random_state=42)
    return shap.TreeExplainer(model, bg)

explainer = get_explainer()

tab1, tab2 = st.tabs(["📝 单例预测","📂 批量预测"])

with tab1:
    st.info("输入10项指标评估术后24h DVT风险")
    with st.form("form"):
        inputs = {}
        n_cols = 3; cols = st.columns(n_cols)
        for i, f in enumerate(features):
            with cols[i % n_cols]:
                if f in CATS:
                    inputs[f] = st.selectbox(LABELS[f], CAT_OPTS[f], index=CAT_OPTS[f].index(DEFAULTS[f]))
                else:
                    inputs[f] = st.number_input(LABELS[f], value=float(DEFAULTS[f]), format="%.2f")
        ok = st.form_submit_button("🚀 评估DVT风险")

    if ok:
        x = pd.DataFrame([inputs], columns=features)
        xs = scaler.transform(x)
        prob = model.predict_proba(xs)[0, 1]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("DVT风险概率", f"{prob*100:.1f}%")
            if prob < 0.15: st.success("🟢 低风险")
            elif prob < 0.35: st.warning("🟡 中风险")
            else: st.error("🔴 高风险")
            if prob > 0.3: st.warning("⚠️ 建议术后超声监测+预防性抗凝")

        with c2:
            st.subheader("🔍 SHAP归因分析")
            with st.spinner("计算特征贡献..."):
                try:
                    sv = explainer(pd.DataFrame(xs, columns=features), check_additivity=False)
                    # Waterfall
                    exp_w = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0],
                                              data=x.iloc[0].values, feature_names=features)
                    fig = plt.figure(figsize=(10, 4))
                    shap.plots.waterfall(exp_w, max_display=10, show=False)
                    plt.title('SHAP Waterfall', fontweight='bold')
                    st.pyplot(fig, bbox_inches='tight'); plt.close()

                    # Force
                    st.markdown("**互动力图** (鼠标悬停查看细节)")
                    force_h = shap.plots.force(sv.base_values[0], np.round(sv.values[0],3),
                                                np.round(x.iloc[0].values,3), feature_names=features,
                                                matplotlib=False)
                    st_shap(force_h, height=160)
                except Exception as e:
                    st.warning(f"SHAP计算失败: {e}")

with tab2:
    st.info("上传含10个特征列的Excel或CSV文件")
    st.code(str(features))
    template = pd.DataFrame(columns=features)
    st.download_button("📥 下载模板", template.to_csv(index=False).encode('utf-8-sig'), "template.csv")

    uf = st.file_uploader("上传文件", type=["xlsx","csv"])
    if uf:
        df_u = pd.read_excel(uf) if uf.name.endswith('.xlsx') else pd.read_csv(uf)
        missing = [c for c in features if c not in df_u.columns]
        if missing:
            st.error(f"缺少列: {missing}")
        else:
            xs_b = scaler.transform(df_u[features])
            probs = model.predict_proba(xs_b)[:, 1]
            df_u['DVT风险'] = probs.round(4)
            df_u['风险等级'] = ['🔴高风险'if p>.35 else'🟡中风险'if p>.15 else'🟢低风险'for p in probs]
            st.dataframe(df_u, use_container_width=True)
            st.download_button("💾 下载结果", df_u.to_csv(index=False).encode('utf-8-sig'), "result.csv")

            st.divider()
            st.subheader("🔍 单样本解释")
            idx = st.selectbox("选择行号", df_u.index)
            if st.button("解释此样本"):
                x_s = df_u.loc[[idx], features]
                xs_s = scaler.transform(x_s)
                sv_s = explainer(pd.DataFrame(xs_s, columns=features), check_additivity=False)
                exp_s = shap.Explanation(values=sv_s.values[0], base_values=sv_s.base_values[0],
                                          data=x_s.iloc[0].values, feature_names=features)
                fig = plt.figure(figsize=(10, 4))
                shap.plots.waterfall(exp_s, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight'); plt.close()
