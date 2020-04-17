import streamlit as st 
import numpy as np 
import pandas as pd 
import base64

st.title('Data analyse deelonderzoek Joris')

st.subheader('Sliders voor de waarde van de correlaties')

# slider voor de cut-off waarde bovenkant
# op dit moment wordt alles weggegooid dat niet belangrijk wordt gezien
cut_off_top = st.slider('cut-off waarde', 0.00 , 1.00 , 0.50)
st.write('cut-off waarde: ', cut_off_top)

# slider voor de onderkant cut-off
# zou nog kunnen worden gebruikt in combinatie met de andere slider
#cut_off_bottom = st.slider('cut-off onderkant', 0.00, 1.00, 0.25)
#st.write('cut-off onderkant: ', cut_off_bottom)

st.subheader('upload je .csv bestand hier')
st.write('let op, het bestand moet een \'utf-8\' codering hebben')

## het uploaden van een .csv bestand 
uploaded_file = st.file_uploader('',type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', index_col=0)
    # kolommen ophalen en die toewijzen aan de dataset gezien de index_col = 0 wordt gebruikt
    # eerste rij eruit te halen
    columns = []
    columns = data.columns
    data.columns = columns
    # de helft weggooien wat dat is niet interessant, de dubbele waardes van een matrix
    data = np.triu(data, k=0)
    data = pd.DataFrame(data=data, columns=columns)
    #data.columns[names]
    data = data.set_index(columns)
    data = data.set_index(columns.T)
    st.write(data)

# wanneer er op de button wordt geklikt, gaat het programmaatje rekenen
if  st.checkbox("pas cut-off toe", False):
    # filter voor alle dezelfde woorden bijv. bakken-bakken
    data_no_1 = data.where(data.le(.99))
    # filter alles eruit onder de slider waarde
    data_higher = data_no_1.where(data_no_1.gt(cut_off_top))
    # vanuit matrix naar unieke rijen en alle rijen weggooien onder de threshold
    unstacked = data_higher.unstack()
    # alle null-waardes worden verwijderd
    unstacked_higher = unstacked[pd.notnull(unstacked)]
    # een dataframe wordt gemaakt
    df_unstacked = unstacked_higher.to_frame()
    # een categorie-kolom wordt gemaakt waarvan de categorie standaard op 0 wordt gezet
    df_unstacked['categorie'] = 0 
    # Laten zien van die tabel
    st.write(df_unstacked)
    # voor alle rijen in de tabel
    if st.checkbox('maak categorieën', False):
        for i in range(len(df_unstacked)):
            
            # neem alle categorieën mee die nog geen categorie hebben
            df_changes = df_unstacked[df_unstacked['categorie'] == 0]
            # vind de naam van het intressepaar (A - B) en splits die op
            string = df_unstacked.iloc[i:].index[0]
            string_1 = string[0]
            string_2 = string[1]
            # kijk naar interesse A en kijk of je die nog ergens anders kan vinden 
            df_changes = df_unstacked.filter(like=string_1, axis = 0).replace(to_replace = 0, value = i+1)
            # voeg een interessepaar met dezelfde naam toe aan deze categorie
            df_unstacked.update(df_changes)
            # doe hetzelfde voor interesse B
            df_changes = df_unstacked.filter(like=string_2, axis = 0).replace(to_replace = 0, value = i+1)
            df_unstacked.update(df_changes)
            
            
            # dit blijft doorgaan totdat alle interesses een categorie hebben

        # het sorteren van de tabel op categorie
        df_download = df_unstacked.sort_values('categorie')
        st.write(df_download)
        # download van csv
        csv = df_unstacked.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)