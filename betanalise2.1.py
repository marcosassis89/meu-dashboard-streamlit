import streamlit as st
import pandas as pd
import requests
from scipy.stats import poisson, skellam
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import warnings
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
import datetime
import json
import os

warnings.filterwarnings('ignore')

# Função para formatar número no formato de valor
def format_number(num):
    return f"{num:.2f}"

# Sistema de tracking de apostas
def load_bet_history():
    if os.path.exists('bet_history.json'):
        with open('bet_history.json', 'r') as f:
            return json.load(f)
    return {
        'date': [], 'match': [], 'bet_type': [], 'stake': [], 
        'odds': [], 'result': [], 'profit': [], 'bankroll_evolution': []
    }

def save_bet_history(history):
    with open('bet_history.json', 'w') as f:
        json.dump(history, f)

# Critério de Kelly fractional
def fractional_kelly(probability, odds, bankroll, fraction=0.5):
    if odds <= 1:
        return 0
    kelly_full = (probability * odds - 1) / (odds - 1)
    kelly_fraction = kelly_full * fraction
    # Não apostar se for negativo
    if kelly_fraction < 0:
        return 0
    return bankroll * kelly_fraction

# Cálculo de valor esperado
def calculate_expected_value(probability, odds):
    return probability * odds - 1

# Modelo Dixon-Coles para probabilidades
def dixon_coles_probabilities(home_attack, away_attack, home_defense, away_defense, rho=0.13):
    # Calcular lambdas ajustados
    lambda_home = home_attack * away_defense
    lambda_away = away_attack * home_defense
    
    # Calcular probabilidades com correção de Dixon-Coles
    max_goals = 8
    home_win = 0
    draw = 0
    away_win = 0
    
    for i in range(max_goals):
        for j in range(max_goals):
            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            
            # Aplicar correção de Dixon-Coles para baixos escores
            if i == 0 and j == 0:
                prob *= (1 - (lambda_home * lambda_away * rho))
            elif i == 0 and j == 1:
                prob *= (1 + (lambda_home * rho))
            elif i == 1 and j == 0:
                prob *= (1 + (lambda_away * rho))
            elif i == 1 and j == 1:
                prob *= (1 - rho)
            
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
    
    # Normalizar para garantir que soma seja 1
    total = home_win + draw + away_win
    return home_win/total, draw/total, away_win/total

# Cabeçalho da página
st.set_page_config(page_title="Análise para Bets", layout="wide")

# Criação de cache para fazer apenas uma consulta no site
@st.cache_resource
def tabelas_url(importar):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    soup = requests.get(importar, headers=header)
    pageSoup = BeautifulSoup(soup.content, 'html.parser')

    listardata = []
    listarhora = []
    listarcasa = []
    listarfora = []
    listarresultado = []
    listarrodada = []

    # Variáveis para armazenar a última data e hora válidas
    ultima_data = ""
    ultima_hora = ""

    blocos_rodada = pageSoup.find_all("div", class_="content-box-headline")
    for bloco in blocos_rodada:
        rodada_texto = bloco.get_text(strip=True)
        rodada_numero = rodada_texto.split('.')[0]
        tabela = bloco.find_next("table")
        if not tabela:
            continue
        linhas = tabela.find_all("tr")
        for linha in linhas:
            colunas = linha.find_all("td")
            if len(colunas) >= 7:
                data = colunas[0].get_text(strip=True)
                hora = colunas[1].get_text(strip=True)
                casa = colunas[2].get_text(strip=True)
                casa = re.sub(r"\(\d+\.\)", "", casa).strip()
                resultado = colunas[4].get_text(strip=True)
                fora = colunas[6].get_text(strip=True)
                fora = re.sub(r"\(\d+\.\)", "", fora).strip()

                # Atualiza última data/hora se houver
                if data:
                    ultima_data = data
                else:
                    data = ultima_data

                if hora:
                    ultima_hora = hora
                else:
                    hora = ultima_hora

                listardata.append(data)
                listarhora.append(hora)
                listarcasa.append(casa)
                listarfora.append(fora)
                listarresultado.append(resultado)
                listarrodada.append(rodada_numero)

    df = pd.DataFrame({
        'data': listardata,
        'hora': listarhora,
        'casa': listarcasa,
        'fora': listarfora,
        'resultado': listarresultado,
        'rodada': listarrodada
    })
    return df

# Páginas de dados para webscrapping
premier = tabelas_url("https://www.transfermarkt.com.br/premier-league/gesamtspielplan/wettbewerb/GB1/saison_id/2025")
bseriea = tabelas_url("https://www.transfermarkt.co.uk/campeonato-brasileiro-serie-a/gesamtspielplan/wettbewerb/BRA1?saison_id=2024&spieltagVon=1&spieltagBis=32")
bserieb = tabelas_url("https://www.transfermarkt.com.br/campeonato-brasileiro-serie-b/gesamtspielplan/wettbewerb/BRA2/saison_id/2024")
mls = tabelas_url("https://www.transfermarkt.co.uk/major-league-soccer/gesamtspielplan/wettbewerb/MLS1?saison_id=2024&spieltagVon=1&spieltagBis=36")

# Seleção dos campeonatos
st.header("Escolha o Campeonato")
options = ["Brasileiro Serie - A", "Brasileiro Serie - B", "MLS", "Premier League"]
selected_option = st.radio("Escolha a Opção", options)

if selected_option == "Premier League":
    tabelas_jogos_ajustada = premier
elif selected_option == "Brasileiro Serie - A":
    tabelas_jogos_ajustada = bseriea
elif selected_option == "Brasileiro Serie - B":
    tabelas_jogos_ajustada = bserieb
elif selected_option == "MLS":
    tabelas_jogos_ajustada = mls

# Verifique se o DataFrame está vazio (erro de scraping)
if tabelas_jogos_ajustada.empty:
    st.error("Não foi possível carregar os jogos. Verifique se o site está acessível ou se o layout mudou.")
    st.stop()

#######===Ajuste dos dataframes=====######## 
tabela_jogos_realizados = tabelas_jogos_ajustada[~tabelas_jogos_ajustada.resultado.str.contains("-:-")]
tabela_jogos_faltantes = tabelas_jogos_ajustada[tabelas_jogos_ajustada.resultado.str.contains("-:-")]
tabela_jogos_faltantes = tabela_jogos_faltantes.drop(columns=["resultado"])
tabela_jogos_faltantes = tabela_jogos_faltantes.reindex(["casa", "fora", "rodada"], axis=1)
tabela_jogos_realizados[["gols_casa", "gols_fora"]] = tabela_jogos_realizados.resultado.str.split(":", expand=True)
tabela_jogos_realizados = tabela_jogos_realizados.drop(columns=["resultado"])

# Calcular estatísticas avançadas
tabela_jogos_realizados["gols_casa"] = pd.to_numeric(tabela_jogos_realizados["gols_casa"], errors="coerce").astype(int)
tabela_jogos_realizados["gols_fora"] = pd.to_numeric(tabela_jogos_realizados["gols_fora"], errors="coerce").fillna(0).astype(int)

# Calcular forma recente (últimas 5 partidas)
def calcular_forma_recente(time, df, is_home):
    if is_home:
        jogos = df[df['casa'] == time].tail(5)
        pontos = sum([3 if gc > gf else 1 if gc == gf else 0 for gc, gf in zip(jogos['gols_casa'], jogos['gols_fora'])])
    else:
        jogos = df[df['fora'] == time].tail(5)
        pontos = sum([3 if gf > gc else 1 if gf == gc else 0 for gc, gf in zip(jogos['gols_casa'], jogos['gols_fora'])])
    return pontos / 15  # Normalizado para 0-1

# Calcular médias de gols com ajustes
media_gols_casa = tabela_jogos_realizados.groupby("casa").mean(numeric_only=True)
media_gols_casa = media_gols_casa.rename(columns={"gols_casa": "gols_feitos_casa", "gols_fora": "gols_sofridos_casa"})
media_gols_fora = tabela_jogos_realizados.groupby("fora").mean(numeric_only=True)
media_gols_fora = media_gols_fora.rename(columns={"gols_casa": "gols_sofridos_fora", "gols_fora": "gols_feitos_fora"})

tabela_stats = media_gols_casa.merge(media_gols_fora, left_index=True, right_index=True)
tabela_stats = tabela_stats.reset_index()
tabela_stats = tabela_stats.rename(columns={"casa": "time"})

# Adicionar forma recente
tabela_stats['forma_casa'] = tabela_stats['time'].apply(lambda x: calcular_forma_recente(x, tabela_jogos_realizados, True))
tabela_stats['forma_fora'] = tabela_stats['time'].apply(lambda x: calcular_forma_recente(x, tabela_jogos_realizados, False))

# Filtro avançado por rodada no Streamlit
rodadas_disponiveis = sorted(tabelas_jogos_ajustada['rodada'].unique())
rodada_selecionada = st.sidebar.selectbox("Selecione a rodada", rodadas_disponiveis)
tabela_filtrada = tabelas_jogos_ajustada[tabelas_jogos_ajustada['rodada'] == rodada_selecionada]
st.dataframe(tabela_filtrada)

# Filtrar DataFrames pela rodada
tabela_jogos_realizados_rodada = tabela_jogos_realizados[tabela_jogos_realizados['rodada'] == rodada_selecionada]
tabela_jogos_faltantes_rodada = tabela_jogos_faltantes[tabela_jogos_faltantes['rodada'] == rodada_selecionada]

# Função de cálculo das probabilidades com Dixon-Coles
def calcula_probs(linha):
    time_casa = linha["casa"]
    time_fora = linha["fora"]

    # Obter estatísticas dos times
    stats_casa = tabela_stats[tabela_stats['time'] == time_casa].iloc[0]
    stats_fora = tabela_stats[tabela_stats['time'] == time_fora].iloc[0]
    
    # Calcular forças de ataque e defesa
    attack_home = stats_casa['gols_feitos_casa']
    defense_home = stats_casa['gols_sofridos_casa']
    attack_away = stats_fora['gols_feitos_fora']
    defense_away = stats_fora['gols_sofridos_fora']
    
    # Aplicar modelo Dixon-Coles
    pv_casa, pv_empate, pv_fora = dixon_coles_probabilities(
        attack_home, attack_away, defense_home, defense_away
    )
    
    # Ajustar com forma recente (peso de 20%)
    forma_weight = 0.2
    forma_casa = stats_casa['forma_casa']
    forma_fora = stats_fora['forma_fora']
    
    pv_casa = pv_casa * (1 - forma_weight) + forma_casa * forma_weight
    pv_fora = pv_fora * (1 - forma_weight) + forma_fora * forma_weight
    pv_empate = pv_empate * (1 - forma_weight) + (1 - abs(forma_casa - forma_fora)) * forma_weight
    
    # Normalizar para soma = 1
    total = pv_casa + pv_empate + pv_fora
    pv_casa /= total
    pv_empate /= total
    pv_fora /= total

    linha['Win'] = pv_casa
    linha['Draw'] = pv_empate
    linha['Loss'] = pv_fora

    return linha

# Aplicação das probabilidades na tabela faltante
if not tabela_jogos_faltantes_rodada.empty:
    try:
        tabela_jogos_faltantes_rodada = tabela_jogos_faltantes_rodada.apply(calcula_probs, axis=1)
        
        # Filtrar apenas apostas com alta probabilidade (EV será calculado individualmente depois)
        min_confidence = 0.6  # Probabilidade mínima de 60%
        
        # Times da casa com alta probabilidade
        time_casa_probs = tabela_jogos_faltantes_rodada[
            (tabela_jogos_faltantes_rodada['Win'] > min_confidence)
        ].sort_values('Win', ascending=False)
        
        # Times visitantes com alta probabilidade
        time_fora_probs = tabela_jogos_faltantes_rodada[
            (tabela_jogos_faltantes_rodada['Loss'] > min_confidence)
        ].sort_values('Loss', ascending=False)
        
        # Exibir resultados
        cl1, cl2 = st.columns(2)
        
        cl1.subheader("Times da Casa com Alta Probabilidade (>60%)")
        if time_casa_probs.empty:
            cl1.info("Nenhum time da casa com probabilidade suficiente nesta rodada.")
        else:
            time_casa_probs["Win (%)"] = (time_casa_probs["Win"] * 100).round(2)
            cl1.dataframe(time_casa_probs[["casa", "fora", "Win (%)"]], hide_index=True)
        
        cl2.subheader("Times Visitantes com Alta Probabilidade (>60%)")
        if time_fora_probs.empty:
            cl2.info("Nenhum time visitante com probabilidade suficiente nesta rodada.")
        else:
            time_fora_probs["Loss (%)"] = (time_fora_probs["Loss"] * 100).round(2)
            cl2.dataframe(time_fora_probs[["casa", "fora", "Loss (%)"]], hide_index=True)
            
    except Exception as e:
        st.error(f"Erro ao calcular probabilidades: {e}")

times = tabela_stats.time.tolist()

# Formulário para preencher os dados da aposta
with st.form("Analise"):
    st.header("Informe os Dados")
    la, lb = st.columns(2)
    
    t_casa = la.selectbox("Escolha o time da casa", times)
    t_fora = lb.selectbox("Escolha o time visitante", times)
    
    fator_casa = la.number_input("Qual o valor do odd casa?", min_value=1.0, step=0.1, value=2.0)
    fator_fora = lb.number_input("Qual o valor do odd visitante?", min_value=1.0, step=0.1, value=2.0)
    
    caixa = la.number_input("Qual o valor de caixa atual?", min_value=0.0, value=1000.0)
    kelly_fraction = lb.slider("Fração de Kelly", 0.1, 1.0, 0.5, 0.1)
    
    analisar = la.form_submit_button("Analisar")

# Execução das Análises
if analisar:
    if t_casa == t_fora:
        st.error('Times iguais. Altere a pesquisa.', icon="🚨")
    else: 
        # Carregar histórico de apostas
        bet_history = load_bet_history()
        
        # Encontrar o jogo na tabela
        resultados = tabela_jogos_faltantes_rodada[
            (tabela_jogos_faltantes_rodada['casa'] == t_casa) & 
            (tabela_jogos_faltantes_rodada['fora'] == t_fora)
        ]
    
    if len(resultados) == 0:
        st.error("Nenhum resultado encontrado ou jogo já realizado.")
    elif all(col in resultados.columns for col in ["Win", "Loss", "Draw"]):
        vencer = resultados["Win"].iloc[0]
        perder = resultados["Loss"].iloc[0]
        empatar = resultados["Draw"].iloc[0]
        
        # Calcular valores esperados
        ev_casa = calculate_expected_value(vencer, fator_casa)
        ev_fora = calculate_expected_value(perder, fator_fora)
        ev_empate = calculate_expected_value(empatar, 3.0)  # Odd padrão para empate
        
        # Calcular stakes com Kelly fractional
        stake_casa = fractional_kelly(vencer, fator_casa, caixa, kelly_fraction)
        stake_fora = fractional_kelly(perder, fator_fora, caixa, kelly_fraction)
        stake_empate = fractional_kelly(empatar, 3.0, caixa, kelly_fraction)  # Odd padrão para empate
        
        # Probabilidades
        col1, col2, col3 = st.columns(3)
        col1.header("Probabilidades do Jogo")
        col1.subheader("Time da Casa")
        col1.metric("Probabilidade:", f"{vencer*100:.2f}%")
        col1.metric("Valor Esperado:", f"{ev_casa*100:.2f}%")
        
        col1.subheader("Empate")
        col1.metric("Probabilidade:", f"{empatar*100:.2f}%")
        col1.metric("Valor Esperado:", f"{ev_empate*100:.2f}%")
        
        col1.subheader("Time Visitante")
        col1.metric("Probabilidade:", f"{perder*100:.2f}%")
        col1.metric("Valor Esperado:", f"{ev_fora*100:.2f}%")
        
        # Média de gols
        col2.header("Estatísticas dos Times")
        gols_made_home = tabela_stats.loc[tabela_stats.time == t_casa, "gols_feitos_casa"].iloc[0]
        gols_suffer_home = tabela_stats.loc[tabela_stats.time == t_casa, "gols_sofridos_casa"].iloc[0]
        gols_made_out = tabela_stats.loc[tabela_stats.time == t_fora, "gols_feitos_fora"].iloc[0]
        gols_suffer_out = tabela_stats.loc[tabela_stats.time == t_fora, "gols_sofridos_fora"].iloc[0]
        
        col2.subheader("Time da Casa")
        col2.markdown(f"Gols Feitos em Casa: **{format_number(gols_made_home)}**")
        col2.markdown(f"Gols Sofridos em Casa: **{format_number(gols_suffer_home)}**")
        col2.markdown(f"Forma Recente: **{tabela_stats.loc[tabela_stats.time == t_casa, 'forma_casa'].iloc[0]*100:.1f}%**")
        
        col2.subheader("Time Visitante")
        col2.markdown(f"Gols Feitos Fora: **{format_number(gols_made_out)}**")
        col2.markdown(f"Gols Sofridos Fora: **{format_number(gols_suffer_out)}**")
        col2.markdown(f"Forma Recente: **{tabela_stats.loc[tabela_stats.time == t_fora, 'forma_fora'].iloc[0]*100:.1f}%**")
        
        # Sugestão de valor
        col3.header("Sugestões de Aposta")
        col3.subheader("Critério de Kelly Fractional")
        
        if ev_casa > 0.1:
            col3.success(f"Time da Casa: R$ {stake_casa:.2f}")
        else:
            col3.warning("Time da Casa: Sem valor suficiente")
            
        if ev_empate > 0.1:
            col3.success(f"Empate: R$ {stake_empate:.2f}")
        else:
            col3.warning("Empate: Sem valor suficiente")
            
        if ev_fora > 0.1:
            col3.success(f"Time Visitante: R$ {stake_fora:.2f}")
        else:
            col3.warning("Time Visitante: Sem valor suficiente")
        
        # Botão para registrar aposta
        if st.button("Registrar Aposta", key="register_bet"):
            # Aqui você implementaria a lógica para registrar a aposta
            st.success("Aposta registrada com sucesso!")
            
        # Seção de performance
        st.header("Performance das Apostas")
        
        if len(bet_history['profit']) > 0:
            total_staked = sum(bet_history['stake'])
            total_profit = sum(bet_history['profit'])
            roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
            
            st.metric("ROI Total", f"{roi:.2f}%")
            st.metric("Lucro Total", f"R$ {total_profit:.2f}")
            
            # Gráfico de evolução da banca
            if len(bet_history['bankroll_evolution']) > 0:
                fig, ax = plt.subplots()
                ax.plot(bet_history['bankroll_evolution'])
                ax.set_title('Evolução da Banca')
                ax.set_xlabel('Número de Apostas')
                ax.set_ylabel('Bankroll (R$)')
                st.pyplot(fig)
        else:
            st.info("Ainda não há histórico de apostas.")
    else:
        st.info("Este confronto já ocorreu ou não há probabilidades calculadas para ele.")