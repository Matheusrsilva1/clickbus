import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações ---
FILEPATH = "df_t.xlsx"  # OU "sua_planilha.xlsx"
# Configurações para os plots (opcional)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Tamanho padrão das figuras

# --- Funções de Análise ---

def load_and_preprocess_data(filepath):
    """Carrega e pré-processa os dados da planilha."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            print("Formato de arquivo não suportado. Use .csv ou .xlsx")
            return None
        print(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

    # Conversão de colunas importantes
    try:
        df['date_purchase'] = pd.to_datetime(df['date_purchase'])
        # Assumindo que 'time_purchase' é uma string como 'HH:MM:SS' ou já um objeto time
        # Se for string, convertemos para datetime para extrair a hora
        # Se der erro aqui, o formato de 'time_purchase' pode precisar de ajuste
        df['time_purchase_hour'] = pd.to_datetime(df['time_purchase'], format='%H:%M:%S', errors='coerce').dt.hour
    except KeyError as e:
        print(f"Erro: Coluna esperada não encontrada: {e}. Verifique os nomes das colunas no seu arquivo.")
        return None
    except Exception as e:
        print(f"Erro ao converter colunas de data/hora: {e}. Verifique os formatos.")
        # Pode ser útil mostrar as primeiras linhas da coluna problemática:
        # print(df[['date_purchase', 'time_purchase']].head())

    # Garantir que colunas numéricas são numéricas
    numeric_cols = ['gmv_success', 'total_tickets_quantity_sucess']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Aviso: Coluna numérica esperada '{col}' não encontrada.")

    # Criar colunas úteis para análise temporal
    if 'date_purchase' in df.columns:
        df['purchase_year_month'] = df['date_purchase'].dt.to_period('M')
        df['purchase_day_of_week'] = df['date_purchase'].dt.day_name()
        df['purchase_month'] = df['date_purchase'].dt.month_name()
        df['purchase_year'] = df['date_purchase'].dt.year

    print("\nPrimeiras 5 linhas após pré-processamento:")
    print(df.head())
    print("\nInformações do DataFrame:")
    df.info()
    print("\nVerificação de valores nulos por coluna:")
    print(df.isnull().sum())

    return df

def descriptive_analysis(df):
    """Realiza a análise descritiva geral."""
    if df is None or 'gmv_success' not in df.columns or 'total_tickets_quantity_sucess' not in df.columns:
        print("Dados insuficientes para análise descritiva.")
        return

    print("\n--- 1. Análise Descritiva Geral ---")
    total_gmv = df['gmv_success'].sum()
    total_tickets = df['total_tickets_quantity_sucess'].sum()
    num_transactions = df['nk_ota_localizer_id'].nunique()
    num_unique_contacts = df['fk_contact'].nunique()

    print(f"GMV Total: R$ {total_gmv:,.2f}")
    print(f"Total de Passagens Vendidas: {total_tickets:,.0f}")
    print(f"Número de Transações Únicas (Localizers): {num_transactions}")
    if num_transactions > 0:
        print(f"GMV Médio por Transação: R$ {total_gmv / num_transactions:,.2f}")
        print(f"Quantidade Média de Passagens por Transação: {total_tickets / num_transactions:,.2f}")
    if total_tickets > 0:
        print(f"Ticket Médio por Passagem: R$ {total_gmv / total_tickets:,.2f}")
    print(f"Período Coberto: de {df['date_purchase'].min().date()} a {df['date_purchase'].max().date()}")
    print(f"Clientes Únicos: {num_unique_contacts}")
    if 'fk_departure_ota_bus_company' in df.columns:
        print(f"Empresas de Ônibus (Ida) Únicas: {df['fk_departure_ota_bus_company'].nunique()}")

def temporal_analysis(df):
    """Realiza a análise temporal e gera gráficos."""
    if df is None or 'purchase_year_month' not in df.columns:
        print("Dados insuficientes para análise temporal.")
        return

    print("\n--- 2. Análise Temporal ---")

    # Vendas por Mês/Ano
    sales_by_month_year = df.groupby('purchase_year_month')['gmv_success'].sum()
    plt.figure()
    sales_by_month_year.plot(kind='line', marker='o')
    plt.title('GMV por Mês/Ano')
    plt.ylabel('GMV (R$)')
    plt.xlabel('Mês/Ano')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("gmv_por_mes_ano.png")
    plt.show()
    print("GMV por Mês/Ano:\n", sales_by_month_year.head())

    # Vendas por Dia da Semana
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sales_by_dow = df.groupby('purchase_day_of_week')['gmv_success'].sum().reindex(days_order)
    plt.figure()
    sales_by_dow.plot(kind='bar')
    plt.title('GMV por Dia da Semana')
    plt.ylabel('GMV (R$)')
    plt.xlabel('Dia da Semana')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("gmv_por_dia_semana.png")
    plt.show()
    print("\nGMV por Dia da Semana:\n", sales_by_dow)

    # Vendas por Hora do Dia
    if 'time_purchase_hour' in df.columns:
        sales_by_hour = df.groupby('time_purchase_hour')['gmv_success'].sum()
        plt.figure()
        sales_by_hour.plot(kind='bar')
        plt.title('GMV por Hora da Compra')
        plt.ylabel('GMV (R$)')
        plt.xlabel('Hora do Dia')
        plt.tight_layout()
        plt.savefig("gmv_por_hora.png")
        plt.show()
        print("\nGMV por Hora da Compra:\n", sales_by_hour.head())

def geographical_analysis(df):
    """Realiza a análise geográfica."""
    if df is None:
        print("Dados insuficientes para análise geográfica.")
        return
    print("\n--- 3. Análise Geográfica ---")

    if 'place_origin_departure' in df.columns:
        top_origins = df['place_origin_departure'].value_counts().nlargest(10)
        plt.figure()
        top_origins.plot(kind='barh')
        plt.title('Top 10 Cidades de Origem (Partida)')
        plt.xlabel('Número de Compras')
        plt.gca().invert_yaxis() # Mostra a mais popular no topo
        plt.tight_layout()
        plt.savefig("top_10_origens.png")
        plt.show()
        print("Top 10 Cidades de Origem (Partida):\n", top_origins)

    if 'place_destination_departure' in df.columns:
        top_destinations = df['place_destination_departure'].value_counts().nlargest(10)
        plt.figure()
        top_destinations.plot(kind='barh')
        plt.title('Top 10 Cidades de Destino (Partida)')
        plt.xlabel('Número de Compras')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("top_10_destinos.png")
        plt.show()
        print("\nTop 10 Cidades de Destino (Partida):\n", top_destinations)

    if 'place_origin_departure' in df.columns and 'place_destination_departure' in df.columns:
        df['route_departure'] = df['place_origin_departure'] + " -> " + df['place_destination_departure']
        top_routes = df['route_departure'].value_counts().nlargest(10)
        plt.figure()
        top_routes.plot(kind='barh')
        plt.title('Top 10 Rotas de Partida Mais Populares')
        plt.xlabel('Número de Compras')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("top_10_rotas.png")
        plt.show()
        print("\nTop 10 Rotas de Partida Mais Populares:\n", top_routes)

def customer_analysis(df):
    """Realiza a análise de clientes."""
    if df is None or 'fk_contact' not in df.columns:
        print("Dados insuficientes para análise de clientes.")
        return
    print("\n--- 4. Análise de Clientes ---")

    purchases_per_customer = df.groupby('fk_contact')['nk_ota_localizer_id'].nunique().sort_values(ascending=False)
    print("Distribuição de Compras por Cliente (Top 10):")
    print(purchases_per_customer.head(10))
    print(f"\nNúmero médio de compras por cliente: {purchases_per_customer.mean():.2f}")
    print(f"Cliente com mais compras: {purchases_per_customer.index[0]} com {purchases_per_customer.iloc[0]} compras.")

    gmv_per_customer = df.groupby('fk_contact')['gmv_success'].sum().sort_values(ascending=False)
    print("\nGMV por Cliente (Top 10):")
    print(gmv_per_customer.head(10).apply(lambda x: f"R$ {x:,.2f}"))
    print(f"Cliente que mais gastou: {gmv_per_customer.index[0]} com R$ {gmv_per_customer.iloc[0]:,.2f}")

def bus_company_analysis(df):
    """Realiza a análise das empresas de ônibus."""
    if df is None or 'fk_departure_ota_bus_company' not in df.columns:
        print("Dados insuficientes para análise de empresas de ônibus.")
        return
    print("\n--- 5. Análise das Empresas de Ônibus ---")

    gmv_by_company_departure = df.groupby('fk_departure_ota_bus_company')['gmv_success'].sum().sort_values(ascending=False).nlargest(10)
    plt.figure()
    gmv_by_company_departure.plot(kind='bar')
    plt.title('Top 10 Empresas de Ônibus (Ida) por GMV')
    plt.ylabel('GMV (R$)')
    plt.xlabel('Empresa de Ônibus (ID)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("top_10_empresas_gmv.png")
    plt.show()
    print("Top 10 Empresas de Ônibus (Ida) por GMV:\n", gmv_by_company_departure.apply(lambda x: f"R$ {x:,.2f}"))

    if 'fk_return_ota_bus_company' in df.columns:
        # Verifica se clientes usam a mesma empresa para ida e volta
        df_ida_volta = df.dropna(subset=['fk_departure_ota_bus_company', 'fk_return_ota_bus_company'])
        if not df_ida_volta.empty:
            same_company_return = (df_ida_volta['fk_departure_ota_bus_company'] == df_ida_volta['fk_return_ota_bus_company']).mean() * 100
            print(f"\nPorcentagem de viagens de ida e volta com a mesma empresa: {same_company_return:.2f}%")

def cart_analysis(df):
    """Realiza a análise do carrinho/compra."""
    if df is None or 'total_tickets_quantity_sucess' not in df.columns:
        print("Dados insuficientes para análise do carrinho.")
        return
    print("\n--- 6. Análise do Carrinho/Compra ---")

    tickets_distribution = df['total_tickets_quantity_sucess'].value_counts().sort_index()
    plt.figure()
    tickets_distribution.plot(kind='bar')
    plt.title('Distribuição da Quantidade de Passagens por Transação')
    plt.ylabel('Número de Transações')
    plt.xlabel('Quantidade de Passagens')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("dist_qtd_passagens.png")
    plt.show()
    print("Distribuição da Quantidade de Passagens por Transação:\n", tickets_distribution)

    # Relação GMV x Quantidade (Preço médio por passagem por transação)
    if 'gmv_success' in df.columns:
        df_valid = df[df['total_tickets_quantity_sucess'] > 0].copy() # Evita divisão por zero
        df_valid['avg_ticket_price_trans'] = df_valid['gmv_success'] / df_valid['total_tickets_quantity_sucess']
        print(f"\nPreço médio por passagem (calculado por transação): R$ {df_valid['avg_ticket_price_trans'].mean():.2f}")
        plt.figure()
        sns.histplot(df_valid['avg_ticket_price_trans'], kde=True, bins=30)
        plt.title('Distribuição do Preço Médio por Passagem por Transação')
        plt.xlabel('Preço Médio por Passagem (R$)')
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.savefig("dist_preco_medio_passagem_transacao.png")
        plt.show()


# --- Execução Principal ---
if __name__ == "__main__":
    # Carrega e pré-processa os dados
    dataframe = load_and_preprocess_data(FILEPATH)

    if dataframe is not None:
        # Realiza as análises
        descriptive_analysis(dataframe)
        temporal_analysis(dataframe)
        geographical_analysis(dataframe)
        customer_analysis(dataframe)
        bus_company_analysis(dataframe)
        cart_analysis(dataframe)

        print("\n\nAnálise concluída. Verifique os outputs no console e os gráficos salvos na pasta do script.")
    else:
        print("Não foi possível carregar os dados. Análise não realizada.")