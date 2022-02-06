###############################################################################
##                              Library imports                              ##
###############################################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import streamlit.components.v1 as components
import requests
import json
import ast
###############################################################################
##                            Functions definition                           ##
###############################################################################
def graph_plot(col, number):
    '''
    This function returns a plot
    '''
    fig1 = plt.figure()
    
    fig1.patch.set_facecolor('#000000')
    fig1.patch.set_alpha(0.1)
    ax = fig1.subplots()  
    sns.kdeplot(data=data[col], color='#4095D1',
                    fill=True, label=col,ax=ax)
    ax.axvline(x=data[col].iloc[number], color='r', linestyle='-')
    ax.set_xlabel(col, fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(False)
    st.sidebar.pyplot(fig1)

def load_models(f):
    '''
    This function loads some data gathered in a pickle file
    '''
    with open(f, 'rb') as file:
        model = pickle.load(file)
    return model

def st_shap(plot, height=None):
    '''
    This function return a shap plot with the html version
    '''
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

###############################################################################
##                     Dashboard parameters definition                       ##
###############################################################################
#    
st.set_page_config(layout="wide") # page expands to full width
sns.set(rc={'axes.facecolor':'F0F2F6', 'figure.facecolor':'F0F2F6'})
#
# Load data
data = pd.read_csv('P7_client_selected_data.csv')
#
###############################################################################
##                              Sidebar definition                           ##
###############################################################################
#    
st.sidebar.header('Client selection') #sidebar title
level = st.sidebar.slider("Select a client number", 1, len(data)+1) 
level = level - 1
#  
st.sidebar.markdown(f"<h2 style='text-align: center';> EXTERNAL SOURCE 1 </h2>", 
                    unsafe_allow_html=True)
graph_plot('EXT_SOURCE_1',level)
#
st.sidebar.markdown(f"<h2 style='text-align: center';> EXTERNAL SOURCE 2 </h2>", 
                    unsafe_allow_html=True)
graph_plot('EXT_SOURCE_2',level)
#
st.sidebar.markdown(f"<h2 style='text-align: center';> EXTERNAL SOURCE 3 </h2>", 
                    unsafe_allow_html=True)
graph_plot('EXT_SOURCE_3',level)
#
st.sidebar.markdown(f"<h2 style='text-align: center';> DAYS EMPLOYED PERCENTAGE</h2>", 
                    unsafe_allow_html=True)
graph_plot('DAYS_EMPLOYED_PERC',level)

#
submit = st.sidebar.button('Get predictions')

st.markdown("<h1 style='text-align: center;'>SCORING MODEL EXPLAINER </h1>", 
            unsafe_allow_html=True)
st.markdown("<h1>  </h1>", unsafe_allow_html=True)

#
###############################################################################
##                                 API definition                            ##
###############################################################################
#    
url = 'https://api-scoring-00.herokuapp.com'
endpoint = '/predict'
#
###############################################################################
##                       Results & Model explaination                        ##
###############################################################################
#    
if submit:
    d =data.iloc[level].to_frame().transpose()
    json_object = json.dumps(d.to_dict('records')[0])
    model = requests.request(method = 'POST', url= url+endpoint, 
                             data = json_object)
    results = ast.literal_eval(model.content.decode("utf-8"))
    prediction = int(results['prediction'])
    probability = float(results['probability'])
    row1_space1, row1_1, row1_2, row1_space3, row1_3, row1_4, row1_space5  = \
    st.columns((0.15, 1.5, 1.0, 0.00000001,1.0, 1.5, 0.15))
    st.markdown(prediction)
    if prediction == 0:
        with row1_2:
            st.markdown("<h2 style='text-align: center; color: black;'> \
                        Loan status : </h2>" , unsafe_allow_html=True)
        with row1_3:
            st.image('approved.png', width = 300)
    elif prediction == 1: 
        with row1_2:
            st.markdown("<h2 style='text-align: center; color: black;'> \
                        Loan status : </h2>" , unsafe_allow_html=True)
        with row1_3:
            st.image('refused.png', width = 300)
    st.markdown(f"<h3 style='text-align: center; color: grey; font-style: italic'> \
                Risk probability: {int(probability*100)} % </h3>" , unsafe_allow_html=True)

    #explainer force_plot
    shap_values, expected_value = load_models('explainer_results.pkl')
    #
    st.markdown("<h3 style='text-align: center; color: black; text-decoration:\
                underline'>  Force plot </strong> </h3>" , unsafe_allow_html=True)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    st_shap(shap.force_plot(expected_value[1], 
                            shap_values[1][level,:], 
                            list(data.columns)))

    row2_space1, row2_1,row2_space2, row2_2, row2_space3 = st.columns(
    (0.15, 1.5, 0.00000001,1.5, 0.15))
    with row2_1:  
        st.markdown("<h3 style='text-align: center; color: black; \
                    text-decoration: underline'>  Summary Plot </h3>" ,
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.summary_plot(shap_values, 
                          data.iloc[level].to_frame().transpose(),plot_type='bar')
        st.pyplot(fig)
#
    with row2_2:
        st.markdown("<h3 style='text-align: center; color: black; \
            text-decoration: underline'>  Decision Plot </h3>" ,
            unsafe_allow_html=True)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.decision_plot(expected_value[1], 
                           shap_values[1][level,:], list(data.columns))
        st.pyplot(fig)
