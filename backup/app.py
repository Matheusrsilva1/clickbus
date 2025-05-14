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
    page_title="Análise de Vendas de Passagens",
    page_icon="🚌",
    layout="wide"
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
    """Análise da Parte 1: Decodificando o Comportamento do Cliente."""
    st.header("Decodificando o Comportamento do Cliente")
    
    # Criar tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral", 
        "👥 Segmentação RFM", 
        "🔄 Padrões de Compra",
        "📈 Análise de Cohort",
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
        
        # Análise de Origens e Destinos
        st.subheader("Principais Origens e Destinos")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'place_origin_departure' in df.columns:
                top_origins = df['place_origin_departure'].value_counts().nlargest(10).reset_index()
                top_origins.columns = ['Cidade', 'Quantidade']
                
                fig = px.bar(top_origins,
                           x='Quantidade',
                           y='Cidade',
                           title='Top 10 Cidades de Origem',
                           orientation='h',
                           color='Quantidade')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'place_destination_departure' in df.columns:
                top_destinations = df['place_destination_departure'].value_counts().nlargest(10).reset_index()
                top_destinations.columns = ['Cidade', 'Quantidade']
                
                fig = px.bar(top_destinations,
                           x='Quantidade',
                           y='Cidade',
                           title='Top 10 Cidades de Destino',
                           orientation='h',
                           color='Quantidade')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Segmentação de Clientes (RFM)")
        
        # Calcular RFM
        rfm = calculate_rfm(df)
        
        # Aplicar segmentação RFM
        rfm_segmented = categorize_rfm(rfm)
        
        # Exibir distribuição dos segmentos
        st.subheader("Distribuição dos Segmentos de Clientes")
        segment_counts = rfm_segmented['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segmento', 'Quantidade']
        
        # Adicionar % do total
        segment_counts['Percentual'] = segment_counts['Quantidade'] / segment_counts['Quantidade'].sum() * 100
        
        fig = px.pie(segment_counts, 
                    values='Quantidade', 
                    names='Segmento', 
                    title='Distribuição de Clientes por Segmento',
                    hole=0.4,
                    color='Segmento',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir valor por segmento
        segment_value = rfm_segmented.groupby('Segment')['monetary'].sum().reset_index()
        segment_value.columns = ['Segmento', 'GMV Total']
        segment_value['Percentual do GMV'] = segment_value['GMV Total'] / segment_value['GMV Total'].sum() * 100
        segment_value['GMV Total'] = segment_value['GMV Total'].round(2)
        segment_value['Percentual do GMV'] = segment_value['Percentual do GMV'].round(2)
        
        fig = px.bar(segment_value, 
                    x='Segmento', 
                    y='GMV Total',
                    title='GMV por Segmento de Cliente',
                    color='Segmento',
                    text='Percentual do GMV')
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise detalhada por segmento
        st.subheader("Perfil dos Segmentos")
        
        # Resumo estatístico por segmento
        segment_profile = rfm_segmented.groupby('Segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'RFM_score': 'mean'
        }).reset_index()
        
        segment_profile.columns = ['Segmento', 'Recência (dias)', 'Frequência', 'Valor (R$)', 'Pontuação RFM']
        segment_profile = segment_profile.round(2)
        
        st.dataframe(segment_profile.sort_values('Pontuação RFM', ascending=False), use_container_width=True)
        
        # Visualização 3D dos segmentos
        fig = px.scatter_3d(rfm_segmented.reset_index().sample(min(1000, len(rfm_segmented))),
                           x='recency', y='frequency', z='monetary',
                           color='Segment',
                           opacity=0.7,
                           title='Visualização 3D dos Segmentos RFM')
        
        fig.update_layout(scene=dict(
            xaxis_title='Recência (dias)',
            yaxis_title='Frequência (compras)',
            zaxis_title='Valor (R$)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Padrões de Compra")
        
        # Análise de Frequência de Compra
        st.subheader("Distribuição de Frequência de Compra")
        purchase_freq = df.groupby('fk_contact').size().reset_index()
        purchase_freq.columns = ['fk_contact', 'Compras']
        
        fig = px.histogram(purchase_freq, 
                          x='Compras',
                          nbins=20,
                          title='Distribuição de Número de Compras por Cliente',
                          labels={'Compras': 'Número de Compras', 'count': 'Quantidade de Clientes'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de Intervalo entre Compras
        st.subheader("Intervalo entre Compras")
        
        # Calcular intervalo entre compras
        df_sorted = df.sort_values(['fk_contact', 'date_purchase'])
        df_sorted['previous_purchase'] = df_sorted.groupby('fk_contact')['date_purchase'].shift(1)
        df_sorted['days_between_purchases'] = (df_sorted['date_purchase'] - df_sorted['previous_purchase']).dt.days
        
        # Filtrar valores válidos
        purchase_intervals = df_sorted.dropna(subset=['days_between_purchases'])
        
        if not purchase_intervals.empty:
            # Limitar a intervalos razoáveis (até 180 dias)
            purchase_intervals = purchase_intervals[purchase_intervals['days_between_purchases'] <= 180]
            
            fig = px.histogram(purchase_intervals, 
                              x='days_between_purchases',
                              nbins=30,
                              title='Distribuição de Dias Entre Compras',
                              labels={'days_between_purchases': 'Dias Entre Compras', 'count': 'Frequência'})
            
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar estatísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média de Dias", f"{purchase_intervals['days_between_purchases'].mean():.1f}")
            with col2:
                st.metric("Mediana de Dias", f"{purchase_intervals['days_between_purchases'].median():.1f}")
            with col3:
                st.metric("Desvio Padrão", f"{purchase_intervals['days_between_purchases'].std():.1f}")
        
        # Análise de ida e volta
        st.subheader("Padrão de Viagens Ida e Volta")
        
        if 'place_origin_return' in df.columns and 'place_destination_return' in df.columns:
            # Percentual de viagens com retorno
            return_pct = df['has_return'].mean() * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("% Viagens com Ida e Volta", f"{return_pct:.1f}%")
            
            with col2:
                # Valor médio de viagens com e sem retorno
                mean_gmv_with_return = df[df['has_return']]['gmv_success'].mean()
                mean_gmv_without_return = df[~df['has_return']]['gmv_success'].mean()
                
                # Diferença percentual
                pct_diff = ((mean_gmv_with_return / mean_gmv_without_return) - 1) * 100 if mean_gmv_without_return > 0 else 0
                
                st.metric("GMV Médio (Ida e Volta)", 
                         f"R$ {mean_gmv_with_return:.2f}", 
                         delta=f"{pct_diff:.1f}% vs. só ida")
            
            # Gráfico de barras para comparação
            comparison_data = pd.DataFrame({
                'Tipo': ['Somente Ida', 'Ida e Volta'],
                'GMV Médio': [mean_gmv_without_return, mean_gmv_with_return],
                'Percentual': [(100 - return_pct), return_pct]
            })
            
            fig = px.bar(comparison_data,
                       x='Tipo',
                       y='GMV Médio',
                       title='Comparação de GMV: Viagens Somente Ida vs. Ida e Volta',
                       color='Tipo',
                       text='Percentual')
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Análise de Cohort")
        
        try:
            # Criar matriz de retenção
            retention_matrix = create_cohort_analysis(df)
            
            # Exibir a matriz de cohort como um heatmap
            st.subheader("Matriz de Retenção por Cohort (% de clientes que retornam)")
            
            # Formatar o índice como string
            retention_matrix.index = retention_matrix.index.astype(str)
            
            # Criar o heatmap com Plotly
            fig = px.imshow(retention_matrix,
                           labels=dict(x="Meses desde primeira compra", y="Cohort (mês de primeira compra)", color="Retenção %"),
                           x=retention_matrix.columns,
                           y=retention_matrix.index,
                           color_continuous_scale='YlGnBu')
            
            fig.update_layout(coloraxis_colorbar=dict(title="% Retenção"))
            
            # Adicionar anotações com os valores
            for i in range(len(retention_matrix.index)):
                for j in range(len(retention_matrix.columns)):
                    if not pd.isna(retention_matrix.iloc[i, j]):
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"{retention_matrix.iloc[i, j]:.1f}%",
                            showarrow=False,
                            font=dict(color="white" if retention_matrix.iloc[i, j] > 50 else "black")
                        )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise de retenção média por período
            st.subheader("Retenção Média por Período")
            
            avg_retention = retention_matrix.mean().reset_index()
            avg_retention.columns = ['Período', 'Retenção Média (%)']
            
            fig = px.line(avg_retention, 
                         x='Período', 
                         y='Retenção Média (%)',
                         markers=True,
                         title='Retenção Média por Período (meses) desde a Primeira Compra')
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erro ao gerar análise de cohort: {str(e)}")
            st.info("A análise de cohort requer dados temporais suficientes para mostrar tendências.")
    
    with tab5:
        st.header("Recomendações Estratégicas")
        
        # Tentar carregar a segmentação RFM, se não estiver carregada
        if 'rfm_segmented' not in locals():
            rfm = calculate_rfm(df)
            rfm_segmented = categorize_rfm(rfm)
        
        # Mostrar recomendações por segmento
        st.write("### Estratégias Recomendadas por Segmento")
        
        segment_strategies = {
            'Campeões': {
                'description': 'Clientes de alto valor que compram com frequência e recentemente.',
                'strategies': [
                    '🌟 Programe de fidelidade exclusivo',
                    '🎁 Descontos especiais para destinos premium',
                    '✉️ Comunicação prioritária sobre novos destinos e serviços',
                    '📊 Monitorar de perto para evitar churn'
                ]
            },
            'Leais': {
                'description': 'Clientes frequentes que não compraram tão recentemente.',
                'strategies': [
                    '🎯 Ofertas "Volte e Ganhe" para estimular nova compra',
                    '👫 Programa "Indique um Amigo"',
                    '🔄 Cupons para aumentar frequência',
                    '🎫 Descontos em destinos já visitados'
                ]
            },
            'Potenciais': {
                'description': 'Compraram recentemente com valor alto, mas pouca frequência.',
                'strategies': [
                    '📆 Incentivos para compras sazonais (feriados, férias)',
                    '🚌 Cross-selling de destinos complementares',
                    '🎟️ Programa de pacotes com desconto progressivo',
                    '📱 Alertas para destinos populares'
                ]
            },
            'Em Risco': {
                'description': 'Bons clientes que não compram há algum tempo.',
                'strategies': [
                    '🚨 Campanha de reativação urgente',
                    '💰 Ofertas substanciais para retorno',
                    '📞 Contato personalizado',
                    '📊 Pesquisa sobre motivos do afastamento'
                ]
            },
            'Hibernando': {
                'description': 'Não compram há tempo e estão ficando inativos.',
                'strategies': [
                    '🔥 Promoções fortes de reativação',
                    '🆕 Comunicação sobre novos serviços e melhorias',
                    '🎯 Ofertas baseadas em histórico anterior',
                    '✉️ Campanha "Sentimos sua Falta"'
                ]
            },
            'Precisam Atenção': {
                'description': 'Recentes, mas baixa frequência e valor.',
                'strategies': [
                    '📈 Incentivos para aumento de ticket médio',
                    '⭐ Comunicação sobre benefícios premium',
                    '🚌 Recomendações personalizadas de rotas',
                    '🎁 Programa de recompensas por gastos'
                ]
            },
            'Novos': {
                'description': 'Primeira compra recente.',
                'strategies': [
                    '👋 Jornada de boas-vindas personalizada',
                    '🎁 Oferta especial para segunda compra',
                    '📱 Incentivo para download do aplicativo',
                    '🔍 Pesquisa de satisfação pós-compra'
                ]
            },
            'Perdidos': {
                'description': 'Baixo engajamento, sem compras recentes.',
                'strategies': [
                    '🔄 Campanha "Última Chance"',
                    '💸 Desconto radical para reativação',
                    '📊 Reduzir frequência de comunicação',
                    '🔎 Análise para entender abandono'
                ]
            }
        }
        
        # Mostrar cards para cada segmento
        segment_columns = st.columns(2)
        
        for i, (segment, data) in enumerate(segment_strategies.items()):
            with segment_columns[i % 2]:
                segment_color = "blue" if segment in ["Campeões", "Leais", "Potenciais", "Novos"] else "orange"
                
                # Calcular número de clientes e valor do segmento
                if segment in rfm_segmented['Segment'].values:
                    segment_size = rfm_segmented[rfm_segmented['Segment'] == segment].shape[0]
                    segment_value = rfm_segmented[rfm_segmented['Segment'] == segment]['monetary'].sum()
                    avg_value = rfm_segmented[rfm_segmented['Segment'] == segment]['monetary'].mean()
                else:
                    segment_size = 0
                    segment_value = 0
                    avg_value = 0
                
                st.markdown(f"""
                <div style="border-left: 5px solid {segment_color}; padding-left: 10px; margin-bottom: 20px;">
                    <h3>{segment}</h3>
                    <p style="color: gray;">{data['description']}</p>
                    <p><b>Clientes:</b> {segment_size:,} | <b>GMV Total:</b> R$ {segment_value:,.2f} | <b>Valor Médio:</b> R$ {avg_value:,.2f}</p>
                    <h4>Estratégias Recomendadas:</h4>
                    <ul>{"".join([f"<li>{strategy}</li>" for strategy in data['strategies']])}</ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Recomendações gerais baseadas na análise
        st.write("### Recomendações Gerais")
        
        # Calcular algumas métricas para recomendações baseadas em dados
        popular_routes = df.groupby(['place_origin_departure', 'place_destination_departure']).size().reset_index()
        popular_routes.columns = ['Origem', 'Destino', 'Contagem']
        top_routes = popular_routes.sort_values('Contagem', ascending=False).head(3)
        
        # Calcular dias da semana mais populares
        popular_days = df.groupby('purchase_day_of_week').size().reset_index()
        popular_days.columns = ['Dia', 'Contagem']
        top_day = popular_days.sort_values('Contagem', ascending=False).iloc[0]['Dia'] if not popular_days.empty else "Segunda"
        top_day_pt = day_map.get(top_day, top_day)
        
        # Calcular horário mais popular
        if 'time_purchase_hour' in df.columns:
            popular_hours = df.groupby('time_purchase_hour').size().reset_index()
            popular_hours.columns = ['Hora', 'Contagem']
            top_hour = popular_hours.sort_values('Contagem', ascending=False).iloc[0]['Hora'] if not popular_hours.empty else 12
        else:
            top_hour = "N/A"
        
        recommendations = [
            {
                'icon': '🚌',
                'title': 'Otimização de Rotas',
                'description': f'Foque campanhas nas rotas mais populares: ' + 
                              ', '.join([f"{r['Origem']} → {r['Destino']}" for _, r in top_routes.iterrows()])
            },
            {
                'icon': '📆',
                'title': 'Campanhas Temporais',
                'description': f'Concentre anúncios promocionais em {top_day_pt}s e por volta das {top_hour}h, horários de maior engajamento'
            },
            {
                'icon': '🔄',
                'title': 'Estratégia de Retenção',
                'description': 'Implemente um programa de comunicação automática após a compra para estimular recompra'
            },
            {
                'icon': '🎯',
                'title': 'Segmentação Avançada',
                'description': 'Desenvolva jornadas específicas para os 8 segmentos identificados, priorizando Campeões e recuperação de clientes Em Risco'
            },
            {
                'icon': '🔍',
                'title': 'Análise de Churn',
                'description': 'Considere um cliente inativo após um período sem compra e implemente estratégias de reativação'
            }
        ]
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"""
            <div style="margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                <h3>{rec['icon']} {rec['title']}</h3>
                <p>{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def parte2_analise(df):
    """Análise da Parte 2: O Timing é Tudo."""
    st.header("O Timing é Tudo")
    st.write("Análise de previsão temporal e classificação binária.")
    
    # Primeiro, calcular RFM
    rfm = calculate_rfm(df)
    
    # Filtrar o DataFrame original para incluir apenas clientes que estão no RFM
    clientes_validos = rfm.index.tolist()
    df_filtered = df[df['fk_contact'].isin(clientes_validos)]
    
    # Ordenar o DataFrame por cliente e data para cálculos corretos de next_purchase
    df_filtered = df_filtered.sort_values(['fk_contact', 'date_purchase'])
    
    # Feature engineering para a Parte 2
    # Calcular dias até a próxima compra para cada cliente
    df_filtered['next_purchase'] = df_filtered.groupby('fk_contact')['date_purchase'].shift(-1)
    df_filtered['days_until_next_purchase'] = (df_filtered['next_purchase'] - df_filtered['date_purchase']).dt.days
    df_filtered['will_buy_next_7_days'] = (df_filtered['days_until_next_purchase'] <= 7).astype(int)
    
    # Remover linhas com valores nulos
    df_clean = df_filtered.dropna(subset=['days_until_next_purchase', 'will_buy_next_7_days'])
    
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
    
    # Agora temos um DataFrame com uma linha por cliente
    X = rfm.loc[df_ultima_compra['fk_contact']][['recency', 'frequency', 'monetary']]
    y = df_ultima_compra['will_buy_next_7_days']
    
    # Verificar se X e y têm o mesmo tamanho
    if len(X) != len(y):
        st.error(f"Erro: X ({len(X)} amostras) e y ({len(y)} amostras) têm tamanhos diferentes.")
        return
    
    # Treinar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.metric("AUC", f"{auc:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")
    
    # Exibir importância das features
    st.subheader("Importância das Features")
    feature_importance = pd.DataFrame({
        'Feature': ['recency', 'frequency', 'monetary'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Feature', y='Importance', title='Importância das Features')
    st.plotly_chart(fig, use_container_width=True)

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
            recency = st.slider("Dias desde última compra", 1, 365, 30)
        with col2:
            frequency = st.slider("Número de compras", 1, 20, 3)
        with col3:
            monetary = st.slider("Valor total gasto (R$)", 100, 5000, 1000)
        
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
    st.title("🚌 Análise de Vendas de Passagens")
    if not os.path.exists(FILEPATH):
        st.error(f"Arquivo '{FILEPATH}' não encontrado na pasta do projeto.")
        return
    df = load_and_preprocess_data(FILEPATH)
    if df is not None:
        st.info(f"Colunas carregadas: {list(df.columns)}")
        # Buscar coluna de tickets de forma tolerante
        tickets_col = find_column(df, ["total_tickets_quantity_sucess", "total_tickets_quantity_success", "tickets", "qtd_passagens"])
        if not tickets_col:
            st.error("Coluna de quantidade de passagens não encontrada. Verifique o nome no arquivo.")
            return
        # Executar diretamente a análise da parte 1
        parte1_analise(df, tickets_col)

if __name__ == "__main__":
    main() 