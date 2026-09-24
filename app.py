import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Clasificación de canciones", page_icon=":musical_note:", layout="wide")


@st.cache_data
def cargar_canciones():
    return pd.read_csv("data/canciones.csv")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelos/modelo_generos.pkl")

df = cargar_canciones()
modelo = cargar_modelo()


df["etiqueta"] = df["track_name"] + " - " + df["artists"]

columnas_modelo = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]

emojis = {
    "classical": "🎻",
    "edm": "🎧",
    "heavy-metal": "🤘",
    "jazz": "🎷",
    "reggaeton": "🔥",
}
colores = {
    "classical": "#F2C14E",
    "edm": "#4CC9F0",
    "heavy-metal": "#F25C54",
    "jazz": "#B388EB",
    "reggaeton": "#1ED760",
}

import streamlit.components.v1 as components

# --- FUNCIONES DE VISUALIZACIÓN ---
def crear_grafico_radar(datos, titulo, color):
    # Elegimos los atributos que van de 0 a 1 para que el gráfico quede equilibrado
    features = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]
    # Extraemos los valores (soporta tanto el DataFrame de búsqueda como el diccionario de creación)
    valores = [datos[f].iloc[0] if isinstance(datos, pd.DataFrame) else datos[f] for f in features]
    
    # Nombres más amigables
    nombres_features = ["Bailabilidad", "Energía", "Positividad", "Acústica", "Instrumental", "En Vivo", "Hablado"]
    
    fig = go.Figure(data=go.Scatterpolar(
      r=valores,
      theta=nombres_features,
      fill='toself',
      marker_color=color,
      line_color=color
    ))
    
    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
      ),
      showlegend=False,
      title=dict(text=titulo, font=dict(size=20), x=0.5),
      margin=dict(l=40, r=40, t=50, b=20),
      height=350
    )
    return fig

# --- INTERFAZ ---
st.title("🎧 ¿Qué género es esta canción?")
st.write("Un modelo de **Machine Learning** analiza el 'ADN sonoro' de la canción y predice su género musical.")

# Métricas principales
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("🎵 Canciones en la Base", f"{len(df):,}")
col2.metric("🗂️ Géneros Posibles", len(df["track_genre"].unique()))
col3.metric("🎯 Acierto del Modelo", "86 %")
st.markdown("---")

tab_buscar, tab_crear, tab_datos = st.tabs(["🔍 Buscar canción", "🎛️ Crear canción (DJ Virtual)", "📊 Ver todas las predicciones"])

with tab_buscar:
    texto = st.text_input("Escribe el nombre de un artista o una canción:", "Avicii")
    
    canciones_encontradas = df[df["etiqueta"].str.contains(texto, case=False, na=False)]
    
    if not canciones_encontradas.empty:
        cancion_seleccionada = st.selectbox("Selecciona una de las coincidencias:", canciones_encontradas["etiqueta"])
        cancion = canciones_encontradas[canciones_encontradas["etiqueta"] == cancion_seleccionada]
        
        col_izq, col_der = st.columns([1, 1.5])
        
        with col_izq:
            track_id = cancion["track_id"].iloc[0]
            components.iframe("https://open.spotify.com/embed/track/" + track_id, height=160)

            genero_real = cancion["track_genre"].iloc[0]
            emoji_real = emojis.get(genero_real, "🎵")
            color_real = colores.get(genero_real, "#1DB954")
            
            st.info(f"**Género Oficial:** {emoji_real} {genero_real.capitalize()}")
            
            # --- MAGIA DEL MODELO AQUÍ ---
            caracteristicas_audio = cancion[columnas_modelo]
            prediccion = modelo.predict(caracteristicas_audio)[0]
            emoji_pred = emojis.get(prediccion, "🤖")
            
            if genero_real == prediccion:
                st.success(f"**El modelo predice:** {emoji_pred} {prediccion.capitalize()} (¡Acertó!)")
            else:
                st.error(f"**El modelo predice:** {emoji_pred} {prediccion.capitalize()} (Se equivocó)")
            
        with col_der:
            # Gráfico de Radar
            fig = crear_grafico_radar(cancion, "Huella Sonora de la Canción", color_real)
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("No se encontraron canciones. Intenta con otro nombre.")

with tab_crear:
    st.markdown("### 🧪 Laboratorio Musical")
    st.write("Juega con los controles deslizantes para crear una canción imaginaria y descubre de qué género sería según la Inteligencia Artificial.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Ritmo y Ánimo**")
        danceability = st.slider("Bailabilidad 💃", 0.0, 1.0, 0.8)
        energy = st.slider("Energía ⚡", 0.0, 1.0, 0.9)
        valence = st.slider("Positividad (Valence) 😃", 0.0, 1.0, 0.7)
        
    with c2:
        st.markdown("**Instrumentación**")
        acousticness = st.slider("Acústica 🎸", 0.0, 1.0, 0.1)
        instrumentalness = st.slider("Instrumentalidad 🎹", 0.0, 1.0, 0.05)
        liveness = st.slider("En vivo (Liveness) 🎤", 0.0, 1.0, 0.2)
        
    with c3:
        st.markdown("**Otros detalles**")
        speechiness = st.slider("Palabras habladas 🗣️", 0.0, 1.0, 0.1)
        loudness = st.slider("Volumen (dB) 🔊", -60.0, 0.0, -5.0)
        tempo = st.slider("Tempo (BPM) ⏱️", 40.0, 220.0, 128.0)
        duration_ms = 200000 # Fijo por simplicidad
        
    # Agrupar los valores seleccionados
    cancion_imaginaria = {
        "danceability": danceability, "energy": energy, "loudness": loudness, 
        "speechiness": speechiness, "acousticness": acousticness, 
        "instrumentalness": instrumentalness, "liveness": liveness, 
        "valence": valence, "tempo": tempo, "duration_ms": duration_ms
    }
    
    # Convertir a DataFrame para el modelo
    df_custom = pd.DataFrame([cancion_imaginaria])
    
    # Predicción
    pred_custom = modelo.predict(df_custom[columnas_modelo])[0]
    emoji_custom = emojis.get(pred_custom, "🎵")
    color_custom = colores.get(pred_custom, "#1DB954")
    
    st.markdown("---")
    res1, res2 = st.columns([1, 1.5])
    
    with res1:
        st.subheader("El modelo dice que tu canción es...")
        st.markdown(f"<h1 style='text-align: center; color: {color_custom}; font-size: 3rem;'>{emoji_custom} {pred_custom.capitalize()}</h1>", unsafe_allow_html=True)
    with res2:
        fig_custom = crear_grafico_radar(cancion_imaginaria, "Tu Huella Sonora", color_custom)
        st.plotly_chart(fig_custom, use_container_width=True)

with tab_datos:
    st.markdown("### 📊 Predicciones para todas las canciones")
    st.write("Explora la tabla para ver qué predijo el modelo para cada canción individual en la base de datos.")
    
    # Calcular las predicciones para todo el dataset
    if "prediccion_modelo" not in df.columns:
        df["prediccion_modelo"] = modelo.predict(df[columnas_modelo])
        # Crear una columna visual para saber si acertó
        df["acierto"] = np.where(df["track_genre"] == df["prediccion_modelo"], "✅ Sí", "❌ No")
        
    # Limpiar y renombrar columnas para que la tabla sea fácil de leer
    columnas_mostrar = ["track_name", "artists", "track_genre", "prediccion_modelo", "acierto", "popularity"]
    df_mostrar = df[columnas_mostrar].rename(columns={
        "track_name": "Canción",
        "artists": "Artista",
        "track_genre": "Género Real",
        "prediccion_modelo": "Predicción del Modelo",
        "acierto": "¿Acertó?",
        "popularity": "Popularidad"
    })
    
    # Controles para filtrar la tabla
    filtro_resultado = st.radio("Filtrar canciones por resultado del modelo:", ["Mostrar Todas", "Solo Aciertos ✅", "Solo Fallos ❌"], horizontal=True)
    
    if filtro_resultado == "Solo Aciertos ✅":
        df_mostrar = df_mostrar[df_mostrar["¿Acertó?"] == "✅ Sí"]
    elif filtro_resultado == "Solo Fallos ❌":
        df_mostrar = df_mostrar[df_mostrar["¿Acertó?"] == "❌ No"]
        
    st.dataframe(df_mostrar, use_container_width=True, height=500)