import streamlit as st,requests,os
st.set_page_config(page_title='MLOps Monitoring',layout='wide')
st.title('Real-Time MLOps Monitoring Dashboard')
api=os.getenv('API_URL','http://localhost:8000')
try:
 info=requests.get(api+'/model/info',timeout=5).json();d=requests.get(api+'/monitor/drift',timeout=5).json()
 a,b,c=st.columns(3);a.metric('Model version',info.get('version','N/A'));b.metric('Validation accuracy',info.get('accuracy','N/A'));c.metric('Drift detected',str(d.get('drift_detected')))
 st.subheader('Drift scores (standardized mean shift)');st.bar_chart(d.get('scores',{}));st.json({'model':info,'drift':d})
except requests.RequestException as e:st.error(f'API unreachable: {e}')
st.subheader('Try a prediction')
with st.form('predict'):
 age=st.number_input('Age',18,120,35);income=st.number_input('Income',0,1000000,60000);tenure=st.number_input('Tenure',0,100,5);usage=st.number_input('Usage',0,10000,45);tickets=st.number_input('Support tickets',0,1000,2)
 if st.form_submit_button('Predict'):
  try:st.json(requests.post(api+'/predict',json={'age':age,'income':income,'tenure':tenure,'usage':usage,'support_tickets':tickets},timeout=10).json())
  except requests.RequestException as e:st.error(str(e))
