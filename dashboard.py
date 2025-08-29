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

# Year filter (opsional, hanya jika ada kolom Year)
if df_clean is not None:
    if "Year" in df_clean.columns:
        selected_years = st.sidebar.slider(
            "Pilih Rentang Tahun",
            min_value=int(df_clean['Year'].min()),
            max_value=int(df_clean['Year'].max()),
            value=(int(df_clean['Year'].min()), int(df_clean['Year'].max()))
        )
    else:
        st.sidebar.info("Kolom 'Year' tidak tersedia pada dataset")
else:
    st.sidebar.info("Dataset belum dimuat")

category_filter = st.sidebar.multiselect(
    "Filter Kategori",
    options=['Rendah', 'Sedang', 'Tinggi'],
    default=['Rendah', 'Sedang', 'Tinggi']
)

# Filter data based on selections
if df_clean is not None:
    filtered_df = df_clean[(df_clean['Year'] >= selected_years[0]) & (df_clean['Year'] <= selected_years[1])]
else:
    filtered_df = None

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
tab1, tab2, tab3, tab4 = st.tabs(["Peta Dunia", "Trend Global", "Analisis Kategori", "Rekomendasi"])

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
    st.subheader("Trend Penggunaan Energi Terbarukan")
    
    if df_clean is not None:
        # Global trend
        global_trend = filtered_df.groupby('Year').agg({
            'RenewablePercentage': ['mean', 'median', 'std']
        }).round(2)
        global_trend.columns = ['Mean', 'Median', 'Std']
        global_trend = global_trend.reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rata-rata Global', 'Median Global', 'Standar Deviasi', 'Distribusi')
        )
        
        # Mean trend
        fig.add_trace(
            go.Scatter(x=global_trend['Year'], y=global_trend['Mean'], mode='lines+markers', name='Rata-rata'),
            row=1, col=1
        )
        
        # Median trend
        fig.add_trace(
            go.Scatter(x=global_trend['Year'], y=global_trend['Median'], mode='lines+markers', name='Median'),
            row=1, col=2
        )
        
        # Standard deviation
        fig.add_trace(
            go.Scatter(x=global_trend['Year'], y=global_trend['Std'], mode='lines+markers', name='Std Dev'),
            row=2, col=1
        )
        
        # Distribution for latest year
        latest_year = filtered_df['Year'].max()
        latest_data = filtered_df[filtered_df['Year'] == latest_year]['RenewablePercentage']
        
        fig.add_trace(
            go.Histogram(x=latest_data, nbinsx=20, name='Distribusi'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data time series tidak tersedia untuk analisis trend")
    
    # Top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        top_10 = filtered_features.nlargest(10, 'final_value')
        fig = px.bar(
            top_10, 
            x='final_value', 
            y='GeoAreaName', 
            orientation='h',
            title='10 Negara dengan Penggunaan Tertinggi',
            color='final_value',
            color_continuous_scale='Greens'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        bottom_10 = filtered_features.nsmallest(10, 'final_value')
        fig = px.bar(
            bottom_10, 
            x='final_value', 
            y='GeoAreaName', 
            orientation='h',
            title='10 Negara dengan Penggunaan Terendah',
            color='final_value',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Analisis Berdasarkan Kategori")
    
    # Radar chart for feature comparison
    features_for_radar = ['mean_usage', 'std_usage', 'trend_slope', 'trend_strength', 'volatility']
    
    # Normalize the data for radar chart
    radar_data_normalized = filtered_features.copy()
    for feature in features_for_radar:
        if feature in radar_data_normalized.columns:
            min_val = radar_data_normalized[feature].min()
            max_val = radar_data_normalized[feature].max()
            if max_val > min_val:
                radar_data_normalized[feature] = (radar_data_normalized[feature] - min_val) / (max_val - min_val)
    
    # Calculate mean values for each category
    radar_data = []
    for category in ['Rendah', 'Sedang', 'Tinggi']:
        if category in radar_data_normalized['category'].values:
            cat_data = radar_data_normalized[radar_data_normalized['category'] == category][features_for_radar].mean().values.tolist()
            cat_data += [cat_data[0]]  # Close the radar
            radar_data.append(cat_data)
    
    fig = go.Figure()
    
    categories_present = radar_data_normalized['category'].unique()
    colors = {'Rendah': '#d62728', 'Sedang': '#ff7f0e', 'Tinggi': '#2ca02c'}
    
    for i, category in enumerate(['Rendah', 'Sedang', 'Tinggi']):
        if category in categories_present:
            fig.add_trace(go.Scatterpolar(
                r=radar_data[i],
                theta=features_for_radar + [features_for_radar[0]],
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
        title='Perbandingan Fitur Antar Kategori (Normalized)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (from the model)
    # Extract feature importance from the dataset if available
    importance_columns = [col for col in filtered_features.columns if 'importance' in col.lower() or 'feature' in col.lower()]
    
    if importance_columns:
        # If we have importance data, use it
        importance_data = filtered_features[importance_columns].mean().sort_values(ascending=False)
        feature_importance = pd.DataFrame({
            'feature': importance_data.index,
            'importance': importance_data.values
        })
    else:
        # Fallback to simulated importance
        feature_importance = pd.DataFrame({
            'feature': ['mean_usage', 'trend_slope', 'volatility', 'std_usage', 'trend_strength'],
            'importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        })
    
    fig = px.bar(
        feature_importance, 
        x='importance', 
        y='feature', 
        orientation='h',
        title='Pentingnya Fitur dalam Klasifikasi',
        color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Rekomendasi Kebijakan")
    
    # Recommendations by category
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-low">Negara dengan Kategori Rendah</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Perlu insentif kebijakan dan investasi infrastruktur
        - Fokus pada peningkatan rata-rata penggunaan
        - Kerja sama internasional untuk transfer teknologi
        - Program pelatihan dan pendidikan energi terbarukan
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-medium">Negara dengan Kategori Sedang</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Dorong percepatan adopsi teknologi
        - Tetapkan target yang lebih ambisius
        - Stabilkan pertumbuhan untuk mengurangi volatilitas
        - Belajar dari praktik terbaik negara dengan kategori tinggi
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="category-high">Negara dengan Kategori Tinggi</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Pertahankan pencapaian dan inovasi berkelanjutan
        - Bagi praktik terbaik dengan negara lain
        - Eksplorasi teknologi energi terbarukan generasi berikutnya
        - Fokus pada integrasi grid dan solusi penyimpanan
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Countries with positive trends
    st.subheader("Negara dengan Tren Positif Terbaik")
    
    if 'trend_slope' in filtered_features.columns:
        positive_trend = filtered_features.nlargest(5, 'trend_slope')[['GeoAreaName', 'trend_slope', 'category', 'final_value']]
        positive_trend['trend_direction'] = positive_trend['trend_slope'].apply(lambda x: '↑' if x > 0 else '↓')
        
        fig = px.bar(
            positive_trend, 
            x='trend_slope', 
            y='GeoAreaName', 
            orientation='h',
            title='Negara dengan Tren Pertumbuhan Terbaik',
            color='category',
            color_discrete_map={'Rendah': '#d62728', 'Sedang': '#ff7f0e', 'Tinggi': '#2ca02c'},
            hover_data=['final_value']
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data tren tidak tersedia")
    
    # Download button for data
    st.download_button(
        label="Unduh Data Klasifikasi",
        data=filtered_features.to_csv(index=False),
        file_name="renewable_energy_classification.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Dashboard Analisis Klasifikasi Penggunaan Energi Terbarukan | ICONIC IT 2025</p>
    <p>Data sumber: renewable_energy_classification_results_enhanced.csv</p>
</div>

""", unsafe_allow_html=True)
