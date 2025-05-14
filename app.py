import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.figure_factory as ff

# Configurações do Streamlit
st.set_page_config(
    page_title="InfiniteBus - ClickBus",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações para os plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

FILEPATH = "df_t.xlsx"  # Caminho fixo

def load_and_preprocess_data(filepath):
    """Carrega e pré-processa os dados da planilha."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            st.error("Formato de arquivo não suportado. Use .csv ou .xlsx")
            return None
        st.success(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas.")
        # Conversão de colunas importantes
        df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')
        n_invalid_dates = df['date_purchase'].isna().sum()
        if n_invalid_dates > 0:
            st.warning(f"{n_invalid_dates} linhas possuem data de compra inválida e foram convertidas para nulo (NaT).")
        df['time_purchase_hour'] = pd.to_datetime(df['time_purchase'], format='%H:%M:%S', errors='coerce').dt.hour
        numeric_cols = ['gmv_success', 'total_tickets_quantity_success']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Criar colunas úteis para análise temporal
        df['purchase_year_month'] = df['date_purchase'].dt.to_period('M')
        df['purchase_day_of_week'] = df['date_purchase'].dt.day_name()
        df['purchase_month'] = df['date_purchase'].dt.month_name()
        df['purchase_year'] = df['date_purchase'].dt.year
        df['purchase_day'] = df['date_purchase'].dt.day
        
        # Criar outras colunas úteis
        df['avg_ticket_price'] = np.where(df['total_tickets_quantity_success'] > 0, 
                                        df['gmv_success'] / df['total_tickets_quantity_success'], 
                                        0)
        
        # Identificar viagens com retorno
        if 'place_origin_return' in df.columns and 'place_destination_return' in df.columns:
            df['has_return'] = (~df['place_origin_return'].isna()) & (df['place_origin_return'] != '0')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None

def find_column(df, possible_names):
    """Procura uma coluna no DataFrame, tolerando pequenas variações de nome."""
    cols = [c.lower().replace(' ', '').replace('_', '') for c in df.columns]
    for name in possible_names:
        name_clean = name.lower().replace(' ', '').replace('_', '')
        for i, col in enumerate(cols):
            if name_clean in col or col in name_clean:
                return df.columns[i]
    return None

def calculate_rfm(df):
    """Calcula as métricas RFM (Recência, Frequência, Monetário) por cliente."""
    rfm = df.groupby('fk_contact').agg({
        'date_purchase': lambda x: (pd.Timestamp.now() - x.max()).days,  # Recência
        'nk_ota_localizer_id': 'count',  # Frequência
        'gmv_success': 'sum'  # Monetário
    }).rename(columns={
        'date_purchase': 'recency',
        'nk_ota_localizer_id': 'frequency',
        'gmv_success': 'monetary'
    })
    return rfm

def parte1_analise(df, tickets_col):
    """InfiniteBus: Mapeamento Comportamental e Segmentação Dinâmica do Cliente."""
    
    # Criar tabs para diferentes análises alinhadas com o InfiniteBus
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral", 
        "👥 Mapeamento Comportamental", 
        "🔄 Projeção de Ciclo de Compra",
        "🗺️ Previsão de Destinos",
        "🎯 Recomendações"
    ])
    
    with tab1:
        st.header("Visão Geral do Negócio")
        
        # Row 1: KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        
        total_gmv = df['gmv_success'].sum()
        total_tickets = df[tickets_col].sum()
        num_transactions = df['nk_ota_localizer_id'].nunique()
        num_customers = df['fk_contact'].nunique()
        
        with col1:
            st.metric("GMV Total", f"R$ {total_gmv:,.2f}")
        with col2:
            st.metric("Total de Passagens", f"{total_tickets:,.0f}")
        with col3:
            st.metric("Transações Únicas", f"{num_transactions:,.0f}")
        with col4:
            st.metric("Clientes Únicos", f"{num_customers:,.0f}")
        
        # Row 2: Métricas derivadas
        col1, col2, col3, col4 = st.columns(4)
        
        avg_gmv = total_gmv / num_transactions if num_transactions > 0 else 0
        avg_tickets = total_tickets / num_transactions if num_transactions > 0 else 0
        avg_ticket_price = total_gmv / total_tickets if total_tickets > 0 else 0
        avg_customer_value = total_gmv / num_customers if num_customers > 0 else 0
        
        with col1:
            st.metric("GMV Médio por Transação", f"R$ {avg_gmv:,.2f}")
        with col2:
            st.metric("Passagens por Transação", f"{avg_tickets:,.2f}")
        with col3:
            st.metric("Ticket Médio por Passagem", f"R$ {avg_ticket_price:,.2f}")
        with col4:
            st.metric("Valor por Cliente", f"R$ {avg_customer_value:,.2f}")
        
        # Análise Temporal
        st.subheader("Análise Temporal de Vendas")
        
        # GMV Mensal com linha de tendência
        st.write("### GMV Mensal")
        sales_by_month_year = df.groupby('purchase_year_month')['gmv_success'].sum().reset_index()
        sales_by_month_year['purchase_year_month'] = sales_by_month_year['purchase_year_month'].astype(str)
        
        # Adicionar linha de tendência
        fig = px.line(sales_by_month_year, 
                    x='purchase_year_month', 
                    y='gmv_success',
                    title='Evolução do GMV Mensal',
                    labels={'gmv_success': 'GMV (R$)', 'purchase_year_month': 'Mês/Ano'})
        
        # Adicionar média móvel de 3 meses como linha de tendência
        if len(sales_by_month_year) >= 3:
            sales_by_month_year['moving_avg'] = sales_by_month_year['gmv_success'].rolling(window=3).mean()
            fig.add_scatter(x=sales_by_month_year['purchase_year_month'], 
                           y=sales_by_month_year['moving_avg'],
                           mode='lines', 
                           name='Média Móvel (3 meses)',
                           line=dict(color='red', width=2, dash='dash'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise por Dia da Semana e Hora
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendas por Dia da Semana
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_names_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            day_map = dict(zip(days_order, day_names_pt))
            
            sales_by_dow = df.groupby('purchase_day_of_week')['gmv_success'].sum().reindex(days_order).reset_index()
            sales_by_dow['day_pt'] = sales_by_dow['purchase_day_of_week'].map(day_map)
            
            fig = px.bar(sales_by_dow,
                       x='day_pt',
                       y='gmv_success',
                       title='GMV por Dia da Semana',
                       labels={'gmv_success': 'GMV (R$)', 'day_pt': 'Dia da Semana'},
                       color='gmv_success')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vendas por Hora do Dia
            if 'time_purchase_hour' in df.columns:
                sales_by_hour = df.groupby('time_purchase_hour')['gmv_success'].sum().reset_index()
                
                fig = px.bar(sales_by_hour,
                           x='time_purchase_hour',
                           y='gmv_success',
                           title='GMV por Hora da Compra',
                           labels={'gmv_success': 'GMV (R$)', 'time_purchase_hour': 'Hora do Dia'},
                           color='gmv_success')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análise RFM (Recência, Frequência, Monetário)")
        
        # Calcular RFM
        rfm = calculate_rfm(df)
        
        # Distribuição de Recência
        st.write("### Distribuição de Recência (dias desde última compra)")
        fig = px.histogram(rfm, x='recency', nbins=30,
                         title='Distribuição de Recência (dias desde última compra)',
                         labels={'recency': 'Dias desde última compra', 'count': 'Quantidade de clientes'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuição de Frequência e Valor
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Distribuição de Frequência")
            fig = px.histogram(rfm, x='frequency', nbins=20,
                             title='Distribuição de Frequência de Compras',
                             labels={'frequency': 'Número de compras', 'count': 'Quantidade de clientes'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Distribuição de Valor Total")
            fig = px.histogram(rfm, x='monetary', nbins=20,
                             title='Distribuição de Valor Total Gasto',
                             labels={'monetary': 'Valor Total Gasto (R$)', 'count': 'Quantidade de clientes'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlações RFM
        st.write("### Correlações entre Recência, Frequência e Valor")
        
        fig = px.scatter(rfm, x='recency', y='frequency', size='monetary',
                       title='Relação entre Recência, Frequência e Valor',
                       labels={'recency': 'Recência (dias)', 'frequency': 'Frequência (compras)', 'monetary': 'Valor (R$)'},
                       opacity=0.6, color='monetary', size_max=30)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Segmentação Dinâmica de Clientes")
        
        # Segmentação RFM
        rfm = calculate_rfm(df)
        rfm_segmented = categorize_rfm(rfm)
        
        # Mostrar contagem de clientes por segmento
        segment_counts = rfm_segmented['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segmento', 'Quantidade']
        
        # Calcular valor total por segmento
        segment_values = rfm_segmented.groupby('Segment')['monetary'].sum().reset_index()
        segment_values.columns = ['Segmento', 'Valor Total']
        
        # Mesclar contagens e valores
        segment_data = pd.merge(segment_counts, segment_values, on='Segmento')
        segment_data['Valor Médio'] = segment_data['Valor Total'] / segment_data['Quantidade']
        segment_data['Percentual de Clientes'] = segment_data['Quantidade'] / segment_data['Quantidade'].sum() * 100
        segment_data['Percentual de Valor'] = segment_data['Valor Total'] / segment_data['Valor Total'].sum() * 100
        
        # Mostrar segmentos em gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Distribuição de Clientes por Segmento")
            segment_data_sorted = segment_data.sort_values('Quantidade', ascending=False)
            
            fig = px.pie(segment_data_sorted, values='Quantidade', names='Segmento',
                       title='Distribuição de Clientes por Segmento',
                       color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Distribuição de Valor por Segmento")
            segment_data_sorted = segment_data.sort_values('Valor Total', ascending=False)
            
            fig = px.pie(segment_data_sorted, values='Valor Total', names='Segmento',
                       title='Distribuição de Valor Total por Segmento',
                       color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com dados dos segmentos
        st.write("### Métricas por Segmento")
        st.dataframe(segment_data[['Segmento', 'Quantidade', 'Percentual de Clientes', 'Valor Total', 'Valor Médio', 'Percentual de Valor']]
                    .sort_values('Valor Total', ascending=False)
                    .reset_index(drop=True)
                    .style.format({
                        'Percentual de Clientes': '{:.1f}%',
                        'Valor Total': 'R$ {:.2f}',
                        'Valor Médio': 'R$ {:.2f}',
                        'Percentual de Valor': '{:.1f}%'
                    }))
        
        # Exibir descrições dos segmentos
        st.write("### Perfil dos Segmentos")
        
        # Carregar descrições dos segmentos do arquivo personas.txt
        try:
            with open('personas.txt', 'r', encoding='utf-8') as file:
                personas = [line.strip() for line in file.readlines() if line.strip()]
                
                # Criar dicionário de personas
                personas_dict = {}
                for persona in personas:
                    if ':' in persona:
                        key, desc = persona.split(':', 1)
                        personas_dict[key.strip()] = desc.strip()
                
                # Exibir em formato de cartões
                for i in range(0, len(personas), 2):
                    col1, col2 = st.columns(2)
                    
                    # Coluna 1
                    if i < len(personas):
                        persona = personas[i]
                        if ':' in persona:
                            segment, description = persona.split(':', 1)
                        else:
                            segment, description = persona, ""
                        
                        with col1:
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                                <h4 style="color: #1E3A8A;">{segment.strip()}</h4>
                                <p>{description.strip()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Coluna 2
                    if i + 1 < len(personas):
                        persona = personas[i + 1]
                        if ':' in persona:
                            segment, description = persona.split(':', 1)
                        else:
                            segment, description = persona, ""
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                                <h4 style="color: #1E3A8A;">{segment.strip()}</h4>
                                <p>{description.strip()}</p>
                            </div>
                            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erro ao carregar descrições dos segmentos: {str(e)}")
    
    with tab4:
        st.subheader("Evolução dos Segmentos ao Longo do Tempo")
        
        # Análise de cohort para entender evolução do comportamento
        try:
            retention_matrix = create_cohort_analysis(df)
            
            # Mostrar matriz de retenção
            st.write("### Matriz de Retenção de Clientes")
            st.write("Percentual de clientes que continuam comprando após sua primeira compra")
            
            # Formatar matriz para exibição
            retention_styled = retention_matrix.style.background_gradient(cmap='Blues')
            retention_styled = retention_styled.format("{:.1f}%")
            
            st.dataframe(retention_styled)
            
            # Visualização da matriz como heatmap
            st.write("### Heatmap de Retenção")
            
            # Criar dataframe adequado para o plotly
            cohort_pivot = retention_matrix.reset_index()
            cohort_pivot.columns = cohort_pivot.columns.astype(str)
            cohort_pivot['primeira_compra'] = cohort_pivot['primeira_compra'].astype(str)
            
            cohort_long = cohort_pivot.melt(
                id_vars=['primeira_compra'],
                var_name='Period',
                value_name='Retention'
            )
            cohort_long['Period'] = cohort_long['Period'].astype(int)
            
            # Plotar heatmap
            fig = px.imshow(
                retention_matrix.values,
                labels=dict(x="Período (meses)", y="Mês da Primeira Compra", color="Retenção (%)"),
                x=[str(i) for i in retention_matrix.columns],
                y=[str(cohort) for cohort in retention_matrix.index],
                color_continuous_scale='Blues',
                zmin=0,
                zmax=100
            )
            fig.update_layout(
                xaxis_title="Meses desde a primeira compra",
                yaxis_title="Cohort (mês da primeira compra)",
                coloraxis_colorbar=dict(title="Retenção (%)")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Curva de retenção média
            st.write("### Curva de Retenção Média")
            avg_retention = retention_matrix.mean(axis=0)
            
            fig = px.line(
                x=avg_retention.index.astype(str),
                y=avg_retention.values,
                markers=True,
                labels={'x': 'Período (meses)', 'y': 'Taxa de Retenção Média (%)'},
                title='Taxa de Retenção Média ao Longo do Tempo'
            )
            
            # Adicionar anotações aos pontos
            for i, val in enumerate(avg_retention.values):
                fig.add_annotation(
                    x=i,
                    y=val,
                    text=f"{val:.1f}%",
                    showarrow=False,
                    yshift=10
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao calcular matriz de retenção: {str(e)}")
            st.info("Esta análise requer dados históricos suficientes para criar cohorts mensais.")
        
        # Dicas de ação baseadas na análise de cohort
        st.write("### Insights e Recomendações")
        
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>Oportunidades Identificadas:</h4>
            <ul>
                <li><strong>Potencial de Reativação</strong>: Clientes que pararam de comprar após o período 3 representam uma oportunidade de reativação.</li>
                <li><strong>Programas de Fidelidade</strong>: Implementar programas focados em clientes que mantêm alta retenção nos primeiros 3 meses.</li>
                <li><strong>Personalização do Ciclo de Vida</strong>: Adaptar comunicações baseadas no estágio do cliente e comportamento histórico de sua cohort.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def parte2_analise(df):
    """Análise da Parte 2: Projeção de Ciclo de Compra e Timing Ideal."""
    st.header("⏱️ Projeção de Ciclo de Compra e Timing Ideal")
    
    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <p>Com base no perfil comportamental estabelecido, o InfiniteBus estima a janela de tempo mais provável 
        para a próxima compra do cliente. Isso considera a frequência individual, recência, intervalo médio entre 
        compras e a sazonalidade das compras anteriores, tanto do indivíduo quanto de seu segmento.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Criar abas para diferentes análises de timing
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "📊 Análise de Intervalo", 
        "🔮 Previsão de Recompra", 
        "📅 Sazonalidade",
        "📱 Simulador de Timing"
    ])
    
    with subtab1:
        st.subheader("Análise de Intervalos entre Compras")
        
        # Ordenar dados por cliente e data
        df_sorted = df.sort_values(['fk_contact', 'date_purchase'])
        
        # Calcular o intervalo entre compras consecutivas para o mesmo cliente
        df_sorted['prev_purchase'] = df_sorted.groupby('fk_contact')['date_purchase'].shift(1)
        df_sorted['days_since_prev_purchase'] = (df_sorted['date_purchase'] - df_sorted['prev_purchase']).dt.days
        
        # Remover valores inválidos ou nulos
        df_intervals = df_sorted.dropna(subset=['days_since_prev_purchase'])
        df_intervals = df_intervals[df_intervals['days_since_prev_purchase'] > 0]
        
        # Mostrar estatísticas de intervalo
        st.write("### Estatísticas do Intervalo entre Compras (dias)")
        
        interval_stats = {
            "Média": df_intervals['days_since_prev_purchase'].mean(),
            "Mediana": df_intervals['days_since_prev_purchase'].median(),
            "Mínimo": df_intervals['days_since_prev_purchase'].min(),
            "Máximo": df_intervals['days_since_prev_purchase'].max(),
            "Desvio Padrão": df_intervals['days_since_prev_purchase'].std()
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Média", f"{interval_stats['Média']:.1f} dias")
        with col2:
            st.metric("Mediana", f"{interval_stats['Mediana']:.1f} dias")
        with col3:
            st.metric("Mínimo", f"{interval_stats['Mínimo']:.1f} dias")
        with col4:
            st.metric("Máximo", f"{interval_stats['Máximo']:.1f} dias")
        with col5:
            st.metric("Desvio Padrão", f"{interval_stats['Desvio Padrão']:.1f} dias")
        
        # Distribuição de intervalos
        st.write("### Distribuição de Intervalos entre Compras")
        
        # Limitar intervalo para visualização (removendo outliers extremos)
        interval_limit = np.percentile(df_intervals['days_since_prev_purchase'], 99)
        df_intervals_viz = df_intervals[df_intervals['days_since_prev_purchase'] <= interval_limit]
        
        fig = px.histogram(
            df_intervals_viz, 
            x='days_since_prev_purchase',
            nbins=50,
            title='Distribuição de Intervalos entre Compras',
            labels={'days_since_prev_purchase': 'Intervalo (dias)', 'count': 'Número de Ocorrências'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Intervalo médio por segmento de cliente
        st.write("### Intervalo Médio por Segmento de Cliente")
        
        # Calcular RFM para segmentação
        rfm = calculate_rfm(df)
        rfm_segmented = categorize_rfm(rfm)
        
        # Mesclar com o dataframe de intervalos
        df_intervals_segment = pd.merge(
            df_intervals,
            rfm_segmented[['Segment']],
            left_on='fk_contact',
            right_index=True,
            how='inner'
        )
        
        # Calcular intervalo médio por segmento
        segment_intervals = df_intervals_segment.groupby('Segment')['days_since_prev_purchase'].agg(['mean', 'median', 'count']).reset_index()
        segment_intervals.columns = ['Segmento', 'Média (dias)', 'Mediana (dias)', 'Quantidade']
        segment_intervals = segment_intervals.sort_values('Média (dias)')
        
        # Visualizar intervalo médio por segmento
        fig = px.bar(
            segment_intervals,
            x='Segmento',
            y='Média (dias)',
            color='Segmento',
            title='Intervalo Médio entre Compras por Segmento',
            text='Média (dias)',
            hover_data=['Mediana (dias)', 'Quantidade']
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with subtab2:
        st.subheader("Previsão de Probabilidade de Recompra")
        
        # Primeiro, calcular RFM
        rfm = calculate_rfm(df)
        
        # Filtrar o DataFrame original para incluir apenas clientes que estão no RFM
        clientes_validos = rfm.index.tolist()
        df_filtered = df[df['fk_contact'].isin(clientes_validos)]
        
        # Ordenar o DataFrame por cliente e data para cálculos corretos de next_purchase
        df_filtered = df_filtered.sort_values(['fk_contact', 'date_purchase'])
        
        # Feature engineering para a previsão de recompra
        # Calcular dias até a próxima compra para cada cliente
        df_filtered['next_purchase'] = df_filtered.groupby('fk_contact')['date_purchase'].shift(-1)
        df_filtered['days_until_next_purchase'] = (df_filtered['next_purchase'] - df_filtered['date_purchase']).dt.days
        
        # Definir janelas de tempo para previsão
        df_filtered['will_buy_next_7_days'] = (df_filtered['days_until_next_purchase'] <= 7).astype(int)
        df_filtered['will_buy_next_15_days'] = (df_filtered['days_until_next_purchase'] <= 15).astype(int)
        df_filtered['will_buy_next_30_days'] = (df_filtered['days_until_next_purchase'] <= 30).astype(int)
        
        # Remover linhas com valores nulos
        df_clean = df_filtered.dropna(subset=['days_until_next_purchase'])
        
        # Agregar dados por cliente para ter uma linha por cliente no final
        cliente_ultima_compra = df_clean.groupby('fk_contact')['date_purchase'].max().reset_index()
        cliente_ultima_compra.columns = ['fk_contact', 'ultima_compra']
        
        # Mesclar com o conjunto limpo para obter apenas a última compra de cada cliente
        df_ultima_compra = pd.merge(
            df_clean, 
            cliente_ultima_compra,
            on='fk_contact',
            how='inner'
        )
        df_ultima_compra = df_ultima_compra[df_ultima_compra['date_purchase'] == df_ultima_compra['ultima_compra']]
        
        try:
            # Agora temos um DataFrame com uma linha por cliente
            X = rfm.loc[df_ultima_compra['fk_contact']][['recency', 'frequency', 'monetary']]
            
            # Escolher qual janela de tempo visualizar
            time_window = st.selectbox(
                "Selecione a janela de tempo para previsão:",
                ['7 dias', '15 dias', '30 dias'],
                index=1
            )
            
            if time_window == '7 dias':
                y = df_ultima_compra['will_buy_next_7_days']
                target_col = 'will_buy_next_7_days'
            elif time_window == '15 dias':
                y = df_ultima_compra['will_buy_next_15_days']
                target_col = 'will_buy_next_15_days'
            else:
                y = df_ultima_compra['will_buy_next_30_days']
                target_col = 'will_buy_next_30_days'
            
            # Verificar se X e y têm o mesmo tamanho
            if len(X) != len(y):
                st.error(f"Erro: X ({len(X)} amostras) e y ({len(y)} amostras) têm tamanhos diferentes.")
            else:
                # Treinar modelo otimizado para memória
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                
                with st.spinner(f'Treinando modelo de previsão para {time_window}...'):
                    model.fit(X_train, y_train)
                
                # Avaliar modelo
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                f1 = f1_score(y_test, y_pred)
                
                # Mostrar métricas
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AUC", f"{auc:.2f}")
                with col2:
                    st.metric("F1 Score", f"{f1:.2f}")
                
                # Exibir importância das features
                st.write("### Importância das Features")
                feature_importance = pd.DataFrame({
                    'Feature': ['Recência (dias)', 'Frequência (compras)', 'Valor Total (R$)'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance, 
                    x='Feature', 
                    y='Importance', 
                    title=f'Importância das Features para Previsão de Recompra em {time_window}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Taxa de conversão por segmento
                st.write(f"### Taxa de Conversão por Segmento (Recompra em {time_window})")
                
                # Mesclar previsões com segmentos
                df_pred = pd.DataFrame({
                    'fk_contact': df_ultima_compra['fk_contact'].values,
                    'actual': y.values,
                    'pred_prob': model.predict_proba(X)[:, 1]
                })
                
                df_pred = pd.merge(
                    df_pred,
                    rfm_segmented[['Segment']],
                    left_on='fk_contact',
                    right_index=True,
                    how='inner'
                )
                
                # Calcular taxa de conversão real por segmento
                segment_conversion = df_pred.groupby('Segment')['actual'].agg(['mean', 'count']).reset_index()
                segment_conversion.columns = ['Segmento', 'Taxa de Conversão', 'Quantidade']
                segment_conversion['Taxa de Conversão'] = segment_conversion['Taxa de Conversão'] * 100
                segment_conversion = segment_conversion.sort_values('Taxa de Conversão', ascending=False)
                
                # Visualizar taxa de conversão por segmento
                fig = px.bar(
                    segment_conversion,
                    x='Segmento',
                    y='Taxa de Conversão',
                    color='Segmento',
                    title=f'Taxa de Conversão por Segmento (Recompra em {time_window})',
                    text='Taxa de Conversão',
                    hover_data=['Quantidade']
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Lista de clientes com maior probabilidade de compra
                st.write("### Top Clientes com Maior Probabilidade de Recompra")
                
                df_pred = df_pred.sort_values('pred_prob', ascending=False)
                top_clients = df_pred.head(10)
                
                # Mesclar com dados RFM
                top_clients_rfm = pd.merge(
                    top_clients,
                    rfm,
                    left_on='fk_contact',
                    right_index=True,
                    how='inner'
                )
                
                # Mostrar tabela de clientes potenciais
                st.dataframe(
                    top_clients_rfm[['fk_contact', 'Segment', 'pred_prob', 'recency', 'frequency', 'monetary']]
                    .rename(columns={
                        'fk_contact': 'ID do Cliente',
                        'Segment': 'Segmento',
                        'pred_prob': 'Probabilidade (%)',
                        'recency': 'Recência (dias)',
                        'frequency': 'Frequência',
                        'monetary': 'Valor Total (R$)'
                    })
                    .style.format({
                        'Probabilidade (%)': '{:.1%}',
                        'Valor Total (R$)': 'R$ {:.2f}'
                    })
                )
        
        except Exception as e:
            st.error(f"Erro ao processar a previsão de recompra: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    with subtab3:
        st.subheader("Análise de Sazonalidade e Padrões Temporais")
        
        # Análise por mês
        st.write("### Sazonalidade Mensal")
        
        # Vendas por mês
        monthly_sales = df.groupby('purchase_month')['gmv_success'].sum().reset_index()
        
        # Ordenar meses corretamente
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_order_pt = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                         'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        month_map = dict(zip(month_order, month_order_pt))
        
        # Mapear nomes dos meses para português e ordenar
        monthly_sales['month_pt'] = monthly_sales['purchase_month'].map(month_map)
        monthly_sales['month_order'] = monthly_sales['purchase_month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
        monthly_sales = monthly_sales.sort_values('month_order')
        
        # Plotar vendas por mês
        fig = px.bar(
            monthly_sales,
            x='month_pt',
            y='gmv_success',
            title='GMV por Mês',
            labels={'gmv_success': 'GMV (R$)', 'month_pt': 'Mês'},
            color='gmv_success'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise por dia da semana e hora
        col1, col2 = st.columns(2)
        
        with col1:
            # Frequência de compras por dia da semana
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_names_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            day_map = dict(zip(days_order, day_names_pt))
            
            purchases_by_dow = df.groupby('purchase_day_of_week').size().reindex(days_order).reset_index()
            purchases_by_dow.columns = ['purchase_day_of_week', 'count']
            purchases_by_dow['day_pt'] = purchases_by_dow['purchase_day_of_week'].map(day_map)
            
            fig = px.bar(
                purchases_by_dow,
                x='day_pt',
                y='count',
                title='Frequência de Compras por Dia da Semana',
                labels={'count': 'Quantidade', 'day_pt': 'Dia da Semana'},
                color='count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Frequência de compras por hora do dia
            if 'time_purchase_hour' in df.columns:
                purchases_by_hour = df.groupby('time_purchase_hour').size().reset_index()
                purchases_by_hour.columns = ['time_purchase_hour', 'count']
                
                fig = px.bar(
                    purchases_by_hour,
                    x='time_purchase_hour',
                    y='count',
                    title='Frequência de Compras por Hora do Dia',
                    labels={'count': 'Quantidade', 'time_purchase_hour': 'Hora do Dia'},
                    color='count'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de dia da semana vs hora
        st.write("### Padrão de Compras: Dia da Semana vs Hora")
        
        if 'time_purchase_hour' in df.columns:
            # Criar heatmap de dia vs hora
            dow_hour = df.groupby(['purchase_day_of_week', 'time_purchase_hour']).size().reset_index()
            dow_hour.columns = ['day', 'hour', 'count']
            
            # Converter para formato adequado para heatmap
            dow_hour_pivot = dow_hour.pivot(index='day', columns='hour', values='count')
            
            # Reordenar dias da semana
            dow_hour_pivot = dow_hour_pivot.reindex(days_order)
            
            # Criar heatmap
            fig = px.imshow(
                dow_hour_pivot.values,
                labels=dict(x="Hora do Dia", y="Dia da Semana", color="Quantidade"),
                x=dow_hour_pivot.columns,
                y=[day_map[day] for day in dow_hour_pivot.index],
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title="Hora do Dia",
                yaxis_title="Dia da Semana",
                coloraxis_colorbar=dict(title="Quantidade")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with subtab4:
        st.subheader("Simulador de Timing Ideal para Contato")
        
        st.markdown("""
        <div style="background-color: #f1f9ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <p>Este simulador permite estimar o melhor momento para contatar um cliente com base em seu perfil comportamental.
            Ajuste os parâmetros para ver a probabilidade de recompra em diferentes janelas de tempo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Primeiro, treinar modelos para diferentes janelas de tempo
            # Usar o mesmo código da subtab2 para preparar os dados
            rfm = calculate_rfm(df)
            clientes_validos = rfm.index.tolist()
            df_filtered = df[df['fk_contact'].isin(clientes_validos)]
            df_filtered = df_filtered.sort_values(['fk_contact', 'date_purchase'])
            
            df_filtered['next_purchase'] = df_filtered.groupby('fk_contact')['date_purchase'].shift(-1)
            df_filtered['days_until_next_purchase'] = (df_filtered['next_purchase'] - df_filtered['date_purchase']).dt.days
            
            df_filtered['will_buy_next_7_days'] = (df_filtered['days_until_next_purchase'] <= 7).astype(int)
            df_filtered['will_buy_next_15_days'] = (df_filtered['days_until_next_purchase'] <= 15).astype(int)
            df_filtered['will_buy_next_30_days'] = (df_filtered['days_until_next_purchase'] <= 30).astype(int)
            
            df_clean = df_filtered.dropna(subset=['days_until_next_purchase'])
            
            cliente_ultima_compra = df_clean.groupby('fk_contact')['date_purchase'].max().reset_index()
            cliente_ultima_compra.columns = ['fk_contact', 'ultima_compra']
            
            df_ultima_compra = pd.merge(df_clean, cliente_ultima_compra, on='fk_contact', how='inner')
            df_ultima_compra = df_ultima_compra[df_ultima_compra['date_purchase'] == df_ultima_compra['ultima_compra']]
            
            X = rfm.loc[df_ultima_compra['fk_contact']][['recency', 'frequency', 'monetary']]
            
            # Verificar se há dados suficientes
            if len(X) < 100:
                st.warning("Poucos dados disponíveis para o simulador. Os resultados podem não ser precisos.")
            
            # Criar modelos para cada janela de tempo
            models = {}
            for window, days in [('7_dias', 7), ('15_dias', 15), ('30_dias', 30)]:
                target = f'will_buy_next_{days}_days'
                y = df_ultima_compra[target]
                
                if len(X) == len(y):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(X_train, y_train)
                    models[window] = model
            
            with st.container():
                st.write("### Configure o Perfil do Cliente")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recency = st.slider("Timing: Dias desde última compra", 1, 365, 35, key="timing_recency_slider")
                with col2:
                    frequency = st.slider("Timing: Número de compras", 1, 20, 4, key="timing_frequency_slider")
                with col3:
                    monetary = st.slider("Timing: Valor total gasto (R$)", 100, 5000, 1500, key="timing_monetary_slider")
                
                # Fazer previsões com os modelos
                client_profile = np.array([[recency, frequency, monetary]])
                
                # Mostrar resultados em uma barra de progresso estilizada
                st.write("### Probabilidade de Recompra")
                
                results = {}
                for window, model in models.items():
                    prob = model.predict_proba(client_profile)[0][1]
                    days = window.split('_')[0]
                    results[f'Próximos {days} dias'] = prob
                
                # Exibir probabilidades como barras de progresso
                for window, prob in results.items():
                    progress_color = "green" if prob > 0.6 else "orange" if prob > 0.3 else "red"
                    progress_html = f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: 500;">{window}</span>
                            <span style="font-weight: 600;">{prob:.1%}</span>
                        </div>
                        <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px;">
                            <div style="width: {prob*100}%; background-color: {progress_color}; height: 10px; border-radius: 10px;"></div>
                        </div>
                    </div>
                    """
                    st.markdown(progress_html, unsafe_allow_html=True)
                
                # Recomendação de timing
                st.write("### Recomendação de Timing")
                
                # Determinar melhor janela de tempo para contato
                if results:
                    best_window = max(results.items(), key=lambda x: x[1])
                    
                    # Calcular dia da semana e hora recomendados com base nos dados
                    if 'time_purchase_hour' in df.columns:
                        # Encontrar o dia e hora com maior frequência de compras
                        dow_hour = df.groupby(['purchase_day_of_week', 'time_purchase_hour']).size().reset_index()
                        dow_hour.columns = ['day', 'hour', 'count']
                        best_time = dow_hour.sort_values('count', ascending=False).iloc[0]
                        
                        best_day_name = day_map.get(best_time['day'], best_time['day'])
                        best_hour = best_time['hour']
                        
                        # Criar recomendação de contato
                        recommendation_html = f"""
                        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 15px;">
                            <h4 style="color: #2e7d32; margin-top: 0;">Recomendação de Contato</h4>
                            <p>Com base no perfil do cliente e nos padrões históricos, recomendamos:</p>
                            <ul>
                                <li><strong>Quando contatar:</strong> Nos {best_window[0].lower()} ({best_window[1]:.1%} de probabilidade)</li>
                                <li><strong>Melhor dia:</strong> {best_day_name}</li>
                                <li><strong>Melhor horário:</strong> Por volta das {best_hour}h</li>
                            </ul>
                            <p style="margin-bottom: 0;"><strong>Estratégia sugerida:</strong> 
                                {
                                    "Oferta agressiva com prazo limitado" if best_window[1] > 0.7 else
                                    "Lembrete personalizado com benefícios" if best_window[1] > 0.4 else
                                    "Comunicação informativa e sem pressão"
                                }
                            </p>
                        </div>
                        """
                        st.markdown(recommendation_html, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Erro ao processar o simulador de timing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def parte3_analise(df):
    """Análise da Parte 3: A Estrada à Frente."""
    st.header("A Estrada à Frente")
    st.write("Análise de classificação multi-classe para recomendação de rotas.")
    
    try:
        # Mostrar as rotas mais populares
        st.subheader("Rotas Mais Populares")
        
        # Feature engineering para criar a coluna de rota
        df['route'] = df['place_origin_departure'] + " -> " + df['place_destination_departure']
        
        # Exibir as rotas mais populares
        route_counts = df['route'].value_counts().reset_index()
        route_counts.columns = ['Rota', 'Frequência']
        top_routes = route_counts.head(10)
        
        fig = px.bar(top_routes, x='Rota', y='Frequência', title='10 Rotas Mais Populares')
        st.plotly_chart(fig, use_container_width=True)
        
        # Usar apenas as N rotas mais populares para evitar problemas de memória
        TOP_N_ROUTES = 10  # Limitar a 10 rotas mais populares
        top_route_names = top_routes['Rota'].tolist()
        
        # Filtrar o DataFrame para incluir apenas as rotas mais populares
        df_top_routes = df[df['route'].isin(top_route_names)].copy()
        
        if len(df_top_routes) == 0:
            st.error("Sem dados suficientes para análise após filtragem das rotas")
            return
            
        # Informação sobre a redução do conjunto de dados
        st.info(f"Análise limitada às {TOP_N_ROUTES} rotas mais populares ({len(df_top_routes)} de {len(df)} registros).")
        
        # Calcular métricas RFM
        # Agrupar por cliente para obter recency, frequency e monetary
        rfm = df_top_routes.groupby('fk_contact').agg({
            'date_purchase': lambda x: (pd.Timestamp.now() - x.max()).days,  # Recência
            'nk_ota_localizer_id': 'count',  # Frequência
            'gmv_success': 'sum'  # Monetário
        }).rename(columns={
            'date_purchase': 'recency',
            'nk_ota_localizer_id': 'frequency',
            'gmv_success': 'monetary'
        })
        
        # Codificar rotas - limitadas às TOP_N mais populares
        le = LabelEncoder()
        df_top_routes['route_encoded'] = le.fit_transform(df_top_routes['route'])
        route_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        # Ordenar o DataFrame por cliente e data para pegar a última rota de cada cliente
        df_top_routes = df_top_routes.sort_values(['fk_contact', 'date_purchase'])
        
        # Pegar a última rota de cada cliente
        last_routes = df_top_routes.groupby('fk_contact').last()[['route', 'route_encoded']]
        
        # Mesclar RFM com as últimas rotas
        modelo_df = rfm.join(last_routes, how='inner')
        
        # Verificar se temos dados suficientes
        if len(modelo_df) < 50:
            st.warning("Poucos dados disponíveis para o modelo. Os resultados podem não ser confiáveis.")
        
        # Prepare X e y para o modelo
        X = modelo_df[['recency', 'frequency', 'monetary']]
        y = modelo_df['route_encoded']
        
        # Amostragem para grandes conjuntos de dados (se necessário)
        MAX_SAMPLES = 5000
        if len(X) > MAX_SAMPLES:
            st.info(f"Amostrando {MAX_SAMPLES} registros dos {len(X)} disponíveis para otimizar o uso de memória.")
            indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]
        
        # Treinar modelo com parâmetros otimizados para memória
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Parâmetros otimizados para usar menos memória
        model = RandomForestClassifier(
            n_estimators=50,           # Menos árvores
            max_depth=10,              # Profundidade limitada
            min_samples_split=10,      # Requer mais amostras para dividir
            min_samples_leaf=5,        # Requer mais amostras por folha
            max_features='sqrt',       # Usa apenas sqrt(n_features) features
            random_state=42,
            n_jobs=-1                  # Usar todos os núcleos da CPU
        )
        
        with st.spinner('Treinando modelo de recomendação de rotas...'):
            model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Métricas de desempenho
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Acurácia do Modelo", f"{accuracy:.2f}")
        with col2:
            n_classes = len(np.unique(y))
            baseline = 1.0 / n_classes if n_classes > 0 else 0
            st.metric("Melhoria sobre Baseline", f"{accuracy/baseline:.1f}x", 
                     delta=f"{(accuracy-baseline)*100:.1f}%")
        
        # Exibir importância das features
        st.subheader("Importância das Features")
        feature_importance = pd.DataFrame({
            'Feature': ['Recência (dias)', 'Frequência (compras)', 'Valor Total (R$)'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Feature', y='Importance', 
                    title='Importância das Features na Recomendação de Rotas')
        st.plotly_chart(fig, use_container_width=True)
        
        # Demonstração de previsão para diferentes perfis de cliente
        st.subheader("Simulador de Recomendação")
        st.write("Ajuste os valores para simular recomendações para diferentes perfis de cliente:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.slider("Rota: Dias desde última compra", 1, 365, 35, key="route_recency_slider")
        with col2:
            frequency = st.slider("Rota: Número de compras", 1, 20, 4, key="route_frequency_slider")
        with col3:
            monetary = st.slider("Rota: Valor total gasto (R$)", 100, 5000, 1500, key="route_monetary_slider")
        
        # Fazer previsão com os valores do slider
        cliente_perfil = np.array([[recency, frequency, monetary]])
        rota_prevista_id = model.predict(cliente_perfil)[0]
        
        # Obter o nome da rota a partir do ID
        rota_prevista_nome = le.inverse_transform([rota_prevista_id])[0]
        
        # Mostrar rota recomendada
        st.success(f"**Rota Recomendada:** {rota_prevista_nome}")
        
        # Mostrar probabilidades para todas as rotas
        probs = model.predict_proba(cliente_perfil)[0]
        # Combinar as rotas com suas probabilidades e ordenar
        routes_probs = [(route, prob) for route, prob in zip(le.classes_, probs)]
        routes_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Mostrar top 3 rotas mais prováveis
        st.write("**Top 3 rotas mais prováveis:**")
        for route, prob in routes_probs[:3]:
            st.write(f"- {route}: {prob*100:.1f}%")
        
    except MemoryError:
        st.error("Erro de memória ao processar os dados. Tente reduzir o conjunto de dados ou otimizar os parâmetros.")
    except Exception as e:
        st.error(f"Erro ao processar análise: {str(e)}")
        st.info("Dica: Verifique se todas as colunas necessárias estão presentes no DataFrame.")
        import traceback
        st.code(traceback.format_exc())

def categorize_rfm(rfm):
    """
    Categoriza os clientes em segmentos com base nas pontuações RFM.
    Usa quartis para criar uma classificação de 1-4 para cada dimensão.
    """
    # Criar quartis para recência (menor é melhor)
    rfm['R_quartile'] = pd.qcut(rfm['recency'], 4, labels=range(4, 0, -1))
    
    # Criar quartis para frequência (maior é melhor)
    rfm['F_quartile'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=range(1, 5))
    
    # Criar quartis para valor monetário (maior é melhor)
    rfm['M_quartile'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=range(1, 5))
    
    # Converter para inteiros para cálculos
    rfm['R_quartile'] = rfm['R_quartile'].astype(int)
    rfm['F_quartile'] = rfm['F_quartile'].astype(int)
    rfm['M_quartile'] = rfm['M_quartile'].astype(int)
    
    # Calcular a pontuação RFM (combinação das três dimensões)
    rfm['RFM_score'] = rfm['R_quartile'] + rfm['F_quartile'] + rfm['M_quartile']
    
    # Segmentar clientes
    rfm['Segment'] = 'Outros'
    
    # Campeões: Clientes que compraram recentemente, compram frequentemente e gastam muito
    rfm.loc[rfm['RFM_score'] >= 10, 'Segment'] = 'Campeões'
    
    # Leais: Compram regularmente, mas não são os mais recentes
    rfm.loc[(rfm['RFM_score'] >= 8) & (rfm['RFM_score'] < 10), 'Segment'] = 'Leais'
    
    # Potenciais: Compraram recentemente, mas não com tanta frequência
    rfm.loc[(rfm['R_quartile'] >= 3) & (rfm['F_quartile'] < 3) & (rfm['M_quartile'] >= 3), 'Segment'] = 'Potenciais'
    
    # Em Risco: Bons clientes que não compram há algum tempo
    rfm.loc[(rfm['R_quartile'] <= 2) & (rfm['F_quartile'] >= 3) & (rfm['M_quartile'] >= 3), 'Segment'] = 'Em Risco'
    
    # Hibernando: Compraram com alguma frequência, mas há muito tempo
    rfm.loc[(rfm['R_quartile'] <= 2) & (rfm['F_quartile'] <= 2) & (rfm['M_quartile'] >= 2), 'Segment'] = 'Hibernando'
    
    # Precisam Atenção: Recentes, mas não gastam muito
    rfm.loc[(rfm['R_quartile'] >= 3) & (rfm['F_quartile'] <= 2) & (rfm['M_quartile'] <= 2), 'Segment'] = 'Precisam Atenção'
    
    # Novos: Compraram muito recentemente pela primeira vez
    rfm.loc[(rfm['R_quartile'] == 4) & (rfm['F_quartile'] == 1), 'Segment'] = 'Novos'
    
    # Perdidos: Não compram há muito tempo e compraram poucas vezes
    rfm.loc[(rfm['R_quartile'] <= 2) & (rfm['F_quartile'] <= 1) & (rfm['M_quartile'] <= 1), 'Segment'] = 'Perdidos'
    
    return rfm

def create_cohort_analysis(df):
    """
    Cria uma análise de cohort com base na primeira compra de cada cliente.
    Analisa retenção ao longo do tempo.
    """
    # Converter data para início do mês
    df['cohort_month'] = df['date_purchase'].dt.to_period('M')
    
    # Identificar a primeira compra de cada cliente
    df_cohort = df.groupby('fk_contact')['cohort_month'].min().reset_index()
    df_cohort.columns = ['fk_contact', 'primeira_compra']
    
    # Mesclar com df original
    df = pd.merge(df, df_cohort, on='fk_contact', how='left')
    
    # Calcular períodos desde a primeira compra
    df['periods'] = (df['cohort_month'].astype(str).astype('period[M]') - 
                    df['primeira_compra'].astype(str).astype('period[M]')).apply(lambda x: int(x))
    
    # Criar matriz de cohort
    cohort_data = df.groupby(['primeira_compra', 'periods'])['nk_ota_localizer_id'].nunique().reset_index()
    cohort_matrix = cohort_data.pivot_table(index='primeira_compra', 
                                          columns='periods', 
                                          values='nk_ota_localizer_id')
    
    # Calcular tamanho de cada cohort (clientes em período 0)
    cohort_sizes = cohort_matrix[0]
    
    # Calcular taxas de retenção
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
    
    return retention_matrix

def main():
    # Aplicar estilo personalizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .sistema-card {
        background-color: #F1F5F9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        color: #4B5563;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<div class="main-header">🚌 InfiniteBus</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema Inteligente de Previsão de Recompra</div>', unsafe_allow_html=True)
    
    # Descrição do sistema em um card
    with st.container():
        st.markdown("""
        <div class="sistema-card">
            <h3>Visão Geral da Solução</h3>
            <p>O InfiniteBus é um sistema inteligente e unificado que utiliza Machine Learning para prever a próxima compra 
            de um cliente da ClickBus. Para alcançar uma previsão completa e acionável, o sistema analisa o comportamento 
            do cliente sob três perspectivas interdependentes, fornecendo um fluxo contínuo de insights para otimizar as vendas.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features do InfiniteBus
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 1.2rem; text-align: center; background-color: #EFF6FF; border-radius: 0.5rem; height: 100%;">
            <div class="feature-icon">👥</div>
            <div class="feature-title">Mapeamento Comportamental</div>
            <div class="feature-description">Criação de perfis dinâmicos que consideram o histórico completo do cliente, 
            afinidades e preferências, indo além do RFM tradicional.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1.2rem; text-align: center; background-color: #EFF6FF; border-radius: 0.5rem; height: 100%;">
            <div class="feature-icon">⏱️</div>
            <div class="feature-title">Projeção de Ciclo de Compra</div>
            <div class="feature-description">Estimativa da janela temporal mais provável para a próxima compra, baseada no 
            perfil comportamental individual e padrões sazonais.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1.2rem; text-align: center; background-color: #EFF6FF; border-radius: 0.5rem; height: 100%;">
            <div class="feature-icon">🗺️</div>
            <div class="feature-title">Identificação do Destino</div>
            <div class="feature-description">Sugestão do trecho com maior probabilidade de ser escolhido, analisando histórico 
            do cliente, comportamento de clientes similares e tendências de mercado.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Carregar dados
    if not os.path.exists(FILEPATH):
        st.error(f"Arquivo '{FILEPATH}' não encontrado na pasta do projeto.")
        return
    
    df = load_and_preprocess_data(FILEPATH)
    if df is not None:
        # Buscar coluna de tickets de forma tolerante
        tickets_col = find_column(df, ["total_tickets_quantity_sucess", "total_tickets_quantity_success", "tickets", "qtd_passagens"])
        if not tickets_col:
            st.error("Coluna de quantidade de passagens não encontrada. Verifique o nome no arquivo.")
            return
            
        # Menu Principal com Tabs
        tab1, tab2, tab3 = st.tabs([
            "🔍 Análise de Comportamento", 
            "⏱️ Previsão de Timing", 
            "🗺️ Recomendação de Destinos"
        ])
        
        with tab1:
            parte1_analise(df, tickets_col)
        
        with tab2:
            parte2_analise(df)
        
        with tab3:
            parte3_analise(df)

if __name__ == "__main__":
    main() 