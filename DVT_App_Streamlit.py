"""CRC术后DVT预测系统 — Streamlit双语版 (10变量LightGBM)"""
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

# ---- Bilingual dictionary (zh / en) ----
T = {
    'app_title': {'zh': '结直肠癌术后DVT风险预测系统', 'en': 'Postoperative DVT Risk Prediction System for Colorectal Cancer'},
    'page_title': {'zh': 'CRC术后DVT预测', 'en': 'CRC Postoperative DVT Prediction'},
    'caption': {'zh': 'LightGBM | LASSO 10变量 | 内部验证AUC 0.866 | 外部验证AUC 0.835',
                'en': 'LightGBM | LASSO 10 variables | Internal Val AUC 0.866 | External Val AUC 0.835'},
    'lang_label': {'zh': '语言 / Language', 'en': 'Language / 语言'},
    'tab_single': {'zh': '📝 单例预测', 'en': '📝 Single Prediction'},
    'tab_batch': {'zh': '📂 批量预测', 'en': '📂 Batch Prediction'},
    'info_input': {'zh': '输入10项指标评估术后DVT风险', 'en': 'Enter the 10 indicators to estimate postoperative DVT risk'},
    'btn_predict': {'zh': '🚀 评估DVT风险', 'en': '🚀 Estimate DVT Risk'},
    'metric_prob': {'zh': 'DVT风险概率', 'en': 'DVT Risk Probability'},
    'low_risk': {'zh': '🟢 低风险', 'en': '🟢 Low risk'},
    'mid_risk': {'zh': '🟡 中风险', 'en': '🟡 Moderate risk'},
    'high_risk': {'zh': '🔴 高风险', 'en': '🔴 High risk'},
    'warn_advice': {'zh': '⚠️ 建议术后超声监测+预防性抗凝', 'en': '⚠️ Postoperative ultrasound surveillance + prophylactic anticoagulation is recommended'},
    'shap_title': {'zh': '🔍 SHAP归因分析', 'en': '🔍 SHAP Attribution Analysis'},
    'spinner_shap': {'zh': '计算特征贡献...', 'en': 'Computing feature contributions...'},
    'force_note': {'zh': '**互动力图** (鼠标悬停查看细节)', 'en': '**Force plot** (hover for details)'},
    'shap_fail': {'zh': 'SHAP计算失败: ', 'en': 'SHAP computation failed: '},
    'info_batch': {'zh': '上传含10个特征列的Excel或CSV文件', 'en': 'Upload an Excel or CSV file containing the 10 feature columns'},
    'dl_template': {'zh': '📥 下载模板', 'en': '📥 Download template'},
    'upload_label': {'zh': '上传文件', 'en': 'Upload file'},
    'missing_cols': {'zh': '缺少列: ', 'en': 'Missing columns: '},
    'col_risk': {'zh': 'DVT风险', 'en': 'DVT Risk'},
    'col_level': {'zh': '风险等级', 'en': 'Risk Level'},
    'dl_result': {'zh': '💾 下载结果', 'en': '💾 Download results'},
    'explain_title': {'zh': '🔍 单样本解释', 'en': '🔍 Single-sample explanation'},
    'select_row': {'zh': '选择行号', 'en': 'Select row index'},
    'btn_explain': {'zh': '解释此样本', 'en': 'Explain this sample'},
    'high_level': {'zh': '🔴高风险', 'en': '🔴High'},
    'mid_level': {'zh': '🟡中风险', 'en': '🟡Moderate'},
    'low_level': {'zh': '🟢低风险', 'en': '🟢Low'},
}

LABELS = {
    'AFR': {'zh': '白蛋白/纤维蛋白原比 AFR', 'en': 'Albumin/fibrinogen ratio (AFR)'},
    'PNI': {'zh': '预后营养指数 PNI', 'en': 'Prognostic nutritional index (PNI)'},
    'Blood_transfusion': {'zh': '围术期输血', 'en': 'Perioperative blood transfusion'},
    'Tumor_site': {'zh': '肿瘤部位', 'en': 'Tumor site'},
    'Varicose_veins': {'zh': '静脉曲张', 'en': 'Varicose veins'},
    'TT': {'zh': '凝血酶时间 TT(秒)', 'en': 'Thrombin time TT (s)'},
    'Age': {'zh': '年龄(岁)', 'en': 'Age (years)'},
    'Operation_time': {'zh': '手术时间(min)', 'en': 'Operation time (min)'},
    'VTE_family_history': {'zh': 'VTE家族史', 'en': 'VTE family history'},
    'D_dimer': {'zh': 'D-二聚体(mg/L)', 'en': 'D-dimer (mg/L)'},
}
CAT_VALUES = {
    'Blood_transfusion': {0: {'zh': '无', 'en': 'No'}, 1: {'zh': '有', 'en': 'Yes'}},
    'Tumor_site': {1: {'zh': '结肠', 'en': 'Colon'}, 2: {'zh': '直肠', 'en': 'Rectum'}},
    'Varicose_veins': {0: {'zh': '无', 'en': 'No'}, 1: {'zh': '有', 'en': 'Yes'}},
    'VTE_family_history': {0: {'zh': '无', 'en': 'No'}, 1: {'zh': '有', 'en': 'Yes'}},
}
DEFAULTS = {'AFR':13.5,'PNI':50.0,'Blood_transfusion':0,'Tumor_site':1,
            'Varicose_veins':0,'TT':13.5,'Age':62,'Operation_time':140,
            'VTE_family_history':0,'D_dimer':3.5}
CATS = ['Blood_transfusion','Tumor_site','Varicose_veins','VTE_family_history']

# ---- Language switch ----
st.set_page_config(page_title=T['page_title']['en'], page_icon="🏥", layout="wide")
lang = st.sidebar.radio(T['lang_label']['zh'], ['中文', 'English'], index=0)
L = 'zh' if lang == '中文' else 'en'

st.title("🏥 " + T['app_title'][L])
st.caption(T['caption'][L])

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

tab1, tab2 = st.tabs([T['tab_single'][L], T['tab_batch'][L]])

with tab1:
    st.info(T['info_input'][L])
    with st.form("form"):
        inputs = {}
        n_cols = 3; cols = st.columns(n_cols)
        for i, f in enumerate(features):
            with cols[i % n_cols]:
                if f in CATS:
                    opts = list(CAT_VALUES[f].keys())
                    opts_lab = [CAT_VALUES[f][v][L] for v in opts]
                    sel = st.selectbox(LABELS[f][L], opts_lab,
                                       index=opts.index(DEFAULTS[f]))
                    inputs[f] = opts[opts_lab.index(sel)]
                else:
                    inputs[f] = st.number_input(LABELS[f][L], value=float(DEFAULTS[f]), format="%.2f")
        ok = st.form_submit_button(T['btn_predict'][L])

    if ok:
        x = pd.DataFrame([inputs], columns=features)
        xs = scaler.transform(x)
        prob = model.predict_proba(xs)[0, 1]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(T['metric_prob'][L], f"{prob*100:.1f}%")
            if prob < 0.15: st.success(T['low_risk'][L])
            elif prob < 0.35: st.warning(T['mid_risk'][L])
            else: st.error(T['high_risk'][L])
            if prob > 0.3: st.warning(T['warn_advice'][L])

        with c2:
            st.subheader(T['shap_title'][L])
            with st.spinner(T['spinner_shap'][L]):
                try:
                    sv = explainer(pd.DataFrame(xs, columns=features), check_additivity=False)
                    exp_w = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0],
                                              data=x.iloc[0].values, feature_names=features)
                    fig = plt.figure(figsize=(10, 4))
                    shap.plots.waterfall(exp_w, max_display=10, show=False)
                    plt.title('SHAP Waterfall', fontweight='bold')
                    st.pyplot(fig, bbox_inches='tight'); plt.close()

                    st.markdown(T['force_note'][L])
                    force_h = shap.plots.force(sv.base_values[0], np.round(sv.values[0],3),
                                                np.round(x.iloc[0].values,3), feature_names=features,
                                                matplotlib=False)
                    st_shap(force_h, height=160)
                except Exception as e:
                    st.warning(T['shap_fail'][L] + str(e))

with tab2:
    st.info(T['info_batch'][L])
    st.code(str(features))
    template = pd.DataFrame(columns=features)
    st.download_button(T['dl_template'][L], template.to_csv(index=False).encode('utf-8-sig'), "template.csv")

    uf = st.file_uploader(T['upload_label'][L], type=["xlsx","csv"])
    if uf:
        df_u = pd.read_excel(uf) if uf.name.endswith('.xlsx') else pd.read_csv(uf)
        missing = [c for c in features if c not in df_u.columns]
        if missing:
            st.error(T['missing_cols'][L] + str(missing))
        else:
            xs_b = scaler.transform(df_u[features])
            probs = model.predict_proba(xs_b)[:, 1]
            df_u[T['col_risk'][L]] = probs.round(4)
            df_u[T['col_level'][L]] = [T['high_level'][L] if p>.35 else T['mid_level'][L] if p>.15 else T['low_level'][L] for p in probs]
            st.dataframe(df_u, use_container_width=True)
            st.download_button(T['dl_result'][L], df_u.to_csv(index=False).encode('utf-8-sig'), "result.csv")

            st.divider()
            st.subheader(T['explain_title'][L])
            idx = st.selectbox(T['select_row'][L], df_u.index)
            if st.button(T['btn_explain'][L]):
                x_s = df_u.loc[[idx], features]
                xs_s = scaler.transform(x_s)
                sv_s = explainer(pd.DataFrame(xs_s, columns=features), check_additivity=False)
                exp_s = shap.Explanation(values=sv_s.values[0], base_values=sv_s.base_values[0],
                                          data=x_s.iloc[0].values, feature_names=features)
                fig = plt.figure(figsize=(10, 4))
                shap.plots.waterfall(exp_s, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight'); plt.close()
