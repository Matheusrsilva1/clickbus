# Análise de Vendas de Passagens 🚌

Este é um dashboard interativo para análise de dados de vendas de passagens de ônibus, desenvolvido com Streamlit.

## Funcionalidades

- 📊 Visão Geral: Métricas principais como GMV total, número de passagens e transações
- 📅 Análise Temporal: Vendas por mês, dia da semana e hora
- 🗺️ Análise Geográfica: Top origens, destinos e rotas
- 👥 Análise de Clientes: Top clientes por número de compras e GMV
- 🚌 Análise de Empresas: Top empresas por GMV e análise de ida e volta
- 🛒 Análise do Carrinho: Distribuição de passagens e preços médios

## Instalação

1. Clone este repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Execute o aplicativo:
```bash
streamlit run app.py
```

2. Faça upload do arquivo Excel (.xlsx) através da interface web
3. Explore as diferentes análises através das abas

## Estrutura do Arquivo de Dados

O arquivo Excel deve conter as seguintes colunas:
- nk_ota_localizer_id: ID da transação
- fk_contact: ID do cliente
- date_purchase: Data da compra
- time_purchase: Hora da compra
- place_origin_departure: Cidade de origem (ida)
- place_destination_departure: Cidade de destino (ida)
- place_origin_return: Cidade de origem (volta)
- place_destination_return: Cidade de destino (volta)
- fk_departure_ota_bus_company: ID da empresa de ônibus (ida)
- fk_return_ota_bus_company: ID da empresa de ônibus (volta)
- gmv_success: Valor total da transação
- total_tickets_quantity_sucess: Quantidade de passagens 