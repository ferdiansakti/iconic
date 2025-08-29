# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Energi Terbarukan Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .category-low {
        color: #d62728;
        font-weight: bold;
    }
    .category-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .category-high {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Load the cleaned and processed data from the notebook
    try:
        # Try to load the enhanced classification results
        df_features = pd.read_csv('renewable_energy_classification_results_enhanced.csv')
        
        # For the time series data, we'll need to reconstruct it from the features
        # or load the original cleaned data if available
        try:
            df_clean = pd.read_csv('renewable_energy_cleaned.csv')
        except:
            # If the cleaned data is not available, create a simplified version
            st.info("Data time series tidak ditemukan. Menampilkan data klasifikasi saja.")
            df_clean = None
            
        return df_clean, df_features
    except FileNotFoundError:
        st.error("File data tidak ditemukan. Pastikan file 'renewable_energy_classification_results_enhanced.csv' tersedia.")
        return None, None

# Load data
df_clean, df_features = load_data()

if df_features is None:
    st.stop()

# Sidebar
st.sidebar.title("🌍 Filter Data")

category_filter = st.sidebar.multiselect(
    "Filter Kategori",
    options=['Rendah', 'Sedang', 'Tinggi'],
    default=['Rendah', 'Sedang', 'Tinggi']
)

# Filter data berdasarkan pilihan
filtered_features = df_features[df_features['category'].isin(category_filter)]

# Main content
st.markdown('<h1 class="main-header">🌍 Klasifikasi Penggunaan Energi Terbarukan</h1>', unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Jumlah Negara", len(filtered_features))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    avg_usage = filtered_features['final_value'].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Rata-rata Penggunaan", f"{avg_usage:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    cat_counts = filtered_features['category'].value_counts()
    
    if 'Rendah' in cat_counts:
        st.markdown(f'<span class="category-low">Rendah: {cat_counts["Rendah"]}</span>', unsafe_allow_html=True)
    if 'Sedang' in cat_counts:
        st.markdown(f'<span class="category-medium">Sedang: {cat_counts["Sedang"]}</span>', unsafe_allow_html=True)
    if 'Tinggi' in cat_counts:
        st.markdown(f'<span class="category-high">Tinggi: {cat_counts["Tinggi"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    max_country = filtered_features.loc[filtered_features['final_value'].idxmax(), 'GeoAreaName']
    max_value = filtered_features['final_value'].max()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Penggunaan Tertinggi", f"{max_value:.1f}%", max_country)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Peta Dunia", "Analisis Kategori", "Perbandingan Negara", "Rekomendasi"])

with tab1:
    st.subheader("Peta Klasifikasi Negara")
    
    # Create a world map
    fig = px.choropleth(
        filtered_features,
        locations="GeoAreaName",
        locationmode="country names",
        color="category",
        hover_name="GeoAreaName",
        hover_data={"final_value": True, "category": True, "mean_usage": True},
        color_discrete_map={'Rendah': '#d62728', 'Sedang': '#ff7f0e', 'Tinggi': '#2ca02c'},
        title="Klasifikasi Negara Berdasarkan Penggunaan Energi Terbarukan"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Analisis Berdasarkan Kategori")
    
    # Radar chart for feature comparison
    features_for_radar = ['mean_usage', 'std_usage', 'trend_slope', 'trend_strength', 'volatility']
    
    # Pastikan fitur-fitur tersebut ada dalam dataset
    available_features = [f for f in features_for_radar if f in filtered_features.columns]
    
    if available_features and len(available_features) > 0:
        # Normalize the data for radar chart
        radar_data_normalized = filtered_features.copy()
        for feature in available_features:
            min_val = radar_data_normalized[feature].min()
            max_val = radar_data_normalized[feature].max()
            if max_val > min_val:
                radar_data_normalized[feature] = (radar_data_normalized[feature] - min_val) / (max_val - min_val)
        
        # Calculate mean values for each category
        radar_data = []
        categories_present = []
        
        for category in ['Rendah', 'Sedang', 'Tinggi']:
            if category in radar_data_normalized['category'].values:
                cat_filtered = radar_data_normalized[radar_data_normalized['category'] == category]
                if len(cat_filtered) > 0:  # Pastikan ada data untuk kategori ini
                    cat_data = cat_filtered[available_features].mean().values.tolist()
                    cat_data += [cat_data[0]]  # Close the radar
                    radar_data.append(cat_data)
                    categories_present.append(category)
        
        # Hanya buat radar chart jika ada data untuk minimal 1 kategori
        if radar_data and len(radar_data) > 0:
            fig = go.Figure()
            
            colors = {'Rendah': '#d62728', 'Sedang': '#ff7f0e', 'Tinggi': '#2ca02c'}
            
            for i, category in enumerate(categories_present):
                if i < len(radar_data):  # Pastikan index tidak melebihi batas
                    fig.add_trace(go.Scatterpolar(
                        r=radar_data[i],
                        theta=available_features + [available_features[0]],
                        fill='toself',
                        name=category,
                        line_color=colors[category]
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title='Perbandingan Fitur Antar Kategori (Normalized)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada data yang cukup untuk membuat radar chart")
    else:
        st.warning("Fitur yang diperlukan untuk radar chart tidak tersedia dalam dataset")
    
    # Feature importance visualization
    # Pilih fitur numerik untuk ditampilkan
    numeric_features = filtered_features.select_dtypes(include=[np.number]).columns.tolist()
    if 'GeoAreaCode' in numeric_features:
        numeric_features.remove('GeoAreaCode')
    
    if numeric_features and len(numeric_features) > 0:
        # Hitung korelasi dengan final_value
        correlations = filtered_features[numeric_features].corrwith(filtered_features['final_value']).abs().sort_values(ascending=False)
        
        # Buat visualisasi
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title='Korelasi dengan Penggunaan Energi Terbarukan',
            labels={'x': 'Korelasi (absolut)', 'y': 'Fitur'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada fitur numerik yang tersedia untuk analisis korelasi")

with tab3:
    st.subheader("Perbandingan Negara")
    
    # Pilih negara untuk dibandingkan
    selected_countries = st.multiselect(
        "Pilih Negara untuk Dibandingkan",
        options=filtered_features['GeoAreaName'].tolist(),
        default=filtered_features.nlargest(3, 'final_value')['GeoAreaName'].tolist() if len(filtered_features) >= 3 else []
    )
    
    if selected_countries and len(selected_countries) > 0:
        comparison_data = filtered_features[filtered_features['GeoAreaName'].isin(selected_countries)]
        
        # Tampilkan metrik utama
        st.subheader("Metrik Utama")
        cols = st.columns(len(selected_countries))
        
        for idx, country in enumerate(selected_countries):
            country_data = comparison_data[comparison_data['GeoAreaName'] == country]
            if len(country_data) > 0:
                country_data = country_data.iloc[0]
                with cols[idx]:
                    st.markdown(f"**{country}**")
                    st.metric("Kategori", country_data['category'])
                    st.metric("Penggunaan", f"{country_data['final_value']:.1f}%")
                    if 'trend_slope' in country_data:
                        st.metric("Tren", f"{country_data['trend_slope']:.2f}")
        
        # Visualisasi perbandingan
        fig = go.Figure()
        
        for country in selected_countries:
            country_data = comparison_data[comparison_data['GeoAreaName'] == country]
            if len(country_data) > 0:
                country_data = country_data.iloc[0]
                
                # Pilih fitur untuk dibandingkan
                features_to_compare = ['mean_usage', 'final_value']
                if 'trend_slope' in country_data:
                    features_to_compare.append('trend_slope')
                
                values = [country_data[f] for f in features_to_compare if f in country_data]
                
                if values:  # Pastikan ada nilai untuk ditampilkan
                    fig.add_trace(go.Bar(
                        name=country,
                        x=features_to_compare,
                        y=values,
                        text=[f"{v:.1f}" for v in values],
                        textposition='auto'
                    ))
        
        if len(fig.data) > 0:  # Pastikan ada data yang ditambahkan
            fig.update_layout(
                title='Perbandingan Fitur Antar Negara',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data yang tersedia untuk visualisasi perbandingan")
    else:
        st.info("Pilih setidaknya satu negara untuk melihat perbandingan")

with tab4:
    st.subheader("Rekomendasi Kebijakan")
    
    # Recommendations by category
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-low">Negara dengan Kategori Rendah</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Insentif Kebijakan**: Berikan subsidi dan tax credit untuk energi terbarukan
        - **Investasi Infrastruktur**: Bangun jaringan smart grid dan fasilitas penyimpanan energi
        - **Kerja Sama Internasional**: Transfer teknologi dari negara maju
        - **Edukasi Publik**: Program kesadaran dan pelatihan teknis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-medium">Negara dengan Kategori Sedang</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Akselerasi Adopsi**: Insentif untuk rooftop solar dan energi distribusi
        - **Target Ambisius**: Tetapkan target 40-50% energi terbarukan pada 2030
        - **Stabilisasi Grid**: Investasi dalam grid modernization dan demand response
        - **Best Practices**: Studi banding ke negara dengan kategori tinggi
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-high">Negara dengan Kategori Tinggi</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Inovasi Berkelanjutan**: R&D untuk teknologi energi terbarukan generasi berikutnya
        - **Ekspor Pengetahuan**: Jadi hub pengetahuan dan pelatihan untuk negara lain
        - **Integrasi Sistem**: Optimasi grid dengan AI dan predictive analytics
        - **Green Hydrogen**: Eksplorasi hydrogen sebagai media penyimpanan energi
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Negara dengan tren terbaik
    if 'trend_slope' in filtered_features.columns and len(filtered_features) > 0:
        st.subheader("Negara dengan Pertumbuhan Tercepat")
        
        positive_trend = filtered_features.nlargest(min(5, len(filtered_features)), 'trend_slope')[['GeoAreaName', 'trend_slope', 'category', 'final_value']]
        
        fig = px.bar(
            positive_trend, 
            x='trend_slope', 
            y='GeoAreaName', 
            orientation='h',
            title='5 Negara dengan Pertumbuhan Tercepat',
            color='category',
            color_discrete_map={'Rendah': '#d62728', 'Sedang': '#ff7f0e', 'Tinggi': '#2ca02c'},
            hover_data=['final_value'],
            labels={'trend_slope': 'Tren Pertumbuhan', 'GeoAreaName': 'Negara'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Download button for data
    if len(filtered_features) > 0:
        st.download_button(
            label="📥 Unduh Data Klasifikasi",
            data=filtered_features.to_csv(index=False),
            file_name="renewable_energy_classification.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Dashboard Analisis Klasifikasi Penggunaan Energi Terbarukan | ICONIC IT 2025</p>
    <p>Dibuat dengan Streamlit • Data: renewable_energy_classification_results_enhanced.csv</p>
</div>
""", unsafe_allow_html=True)
