import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import LabelSet, Label
import umap 
import umap.plot 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

@st.cache
def getUmapEmbedding(data):
    umap_embedding = umap.UMAP(n_components = 2, metric='correlation').fit(data)

    return umap_embedding



## the actual app ##
st.markdown('# plots en analyse #')
st.markdown('__testversie__ 0.1')

## file uploader ##
st.markdown(' ### upload hieronder je .csv bestand ###')
uploaded_file = st.file_uploader('',type="csv")
if uploaded_file is not None:
    with st.spinner('data verwerken'):
            data = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1', delimiter=';')
    
    #test = rob_test_df.drop(['interesses'], axis=1)
    data_cleaned = data.drop(data.columns[0], axis=1)
    
    st.dataframe(data_cleaned)
    data_without_names = data_cleaned.drop(['Interest_Name'], axis= 1)
    data_labels = data_cleaned['Interest_Name'].astype('category')
st.markdown('## PCA (het terugbrengen van het groot aantal dimensies) ##')

if data is not None:
    n_components = st.number_input(label='aantal dimensies', min_value=1, max_value= int(data_cleaned.count(axis= 1)[0]), value= 2)

    start_computation = st.button(label= 'start berekening')
    if start_computation == True:
        
        pca = PCA(n_components= n_components)
        pca_array = pca.fit_transform(data_without_names)
        st.write(pca_array)
        
        if pca_array.shape[1] == 2:
            pca_df = pd.DataFrame(pca_array, columns=['x','y'])
            data_df = pd.DataFrame(data=data_cleaned['Interest_Name'])
            pca_plot_ready = data_df.join(pca_df)
            st.write(pca_plot_ready)

            ############## bokeh plot info ######################
            p = figure(title = 'pca plot')

            p.circle(pca_plot_ready['x'], pca_plot_ready['y'])

            labels = LabelSet(x='x', y='y', text='Interest_Name', level='glyph',
            x_offset=5, y_offset=5, text_font_size = '8pt', source= ColumnDataSource(pca_plot_ready), render_mode='canvas')

            output_file('PCA.html', title='PCA plot', mode='inline')

            p.add_layout(labels)

            ############# bokeh plot info end ######################
            st.bokeh_chart(p)

    st.markdown(' # UMAP kijken wat er gebeurd # ')
    start_umap = st.button(label= 'start umap berekening')
    if start_umap == True:
        umap_embedding = getUmapEmbedding(data_without_names)
        st.write('done with the umap embedding')
        f = umap.plot.interactive(umap_embedding, labels=data_labels, hover_data=data_labels, point_size=10)
        st.bokeh_chart(f)

        st.write(umap_embedding)

    st.markdown(' # similarity index  # ')

    np_data = data_without_names.to_numpy() 
    matrix = cosine_similarity(np_data, np_data)
    corr = pd.DataFrame(data= matrix)
    sns.set(style= 'white')
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    halve_matrix = corr.rename(columns= data_cleaned['Interest_Name'])
    hele_matrix = halve_matrix.set_index(data_cleaned['Interest_Name'].T)

    st.write(hele_matrix)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    similarity_plot = sns.heatmap(hele_matrix, mask=mask, cmap=cmap, vmax=1, vmin=0, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    similarity_plot.set_xticklabels(similarity_plot.get_xticklabels(), rotation=45, horizontalalignment= 'right')

    st.pyplot()