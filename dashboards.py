import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import io
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# Atualização forçada para commit

# === Carregar dados ===
df = pd.read_excel('data_raw/saida.xlsx', sheet_name='Crescimento (%)')
df['Data'] = pd.to_datetime(df['Data']).dt.date  # remove hora

# Não precisa criar a coluna 'Tamanho MB', pois já existe 'Tamanho (MB)'

# === Corrigir cálculo de crescimento percentual ===
df = df.sort_values(['Base', 'Data'])
def calcular_crescimento_percentual(grupo):
    grupo = grupo.sort_values('Data').copy()
    tamanhos = grupo['Tamanho (MB)'].values
    crescimento = [0.0]
    for i in range(1, len(tamanhos)):
        anterior = tamanhos[i - 1]
        atual = tamanhos[i]
        variação = ((atual - anterior) / anterior) * 100 if anterior != 0 else 0
        crescimento.append(variação)
    grupo['Crescimento (%)'] = crescimento
    return grupo

df = df.groupby('Base', group_keys=False).apply(calcular_crescimento_percentual)

# === Sidebar ===
st.sidebar.title("🔎 Filtros")
bases_disponiveis = sorted(df['Base'].unique())
base_padrao = bases_disponiveis[:1]
bases_selecionadas = st.sidebar.multiselect(
    "Selecione as Bases", bases_disponiveis, default=base_padrao
)
data_max = df['Data'].max()
data_min_padrao = data_max.replace(year=data_max.year - 1)
periodo = st.sidebar.date_input(
    "Escolha o intervalo de datas",
    value=(data_min_padrao, data_max),
    min_value=df['Data'].min(),
    max_value=data_max
)
inicio = pd.to_datetime(periodo[0])
fim = pd.to_datetime(periodo[1])

# === Filtrar dados ===
df_filtrado = df[
    df['Base'].isin(bases_selecionadas) &
    (pd.to_datetime(df['Data']) >= inicio) &
    (pd.to_datetime(df['Data']) <= fim)
]

# === Título ===
st.title("📊 Dashboard de Crescimento das Bases de Dados")
st.write("Acompanhe a evolução, projeções e variações das bases selecionadas.")

# === Alerta automático ===
if any(df_filtrado['Crescimento (%)'] > 50):
    st.warning("🚨 Algumas bases tiveram crescimento acima de 50%!")

# === Gráfico com suavização e tendência polinomial ===
st.subheader("📈 Evolução do Tamanho com Suavização e Tendência")

df_suave = df_filtrado.copy()
df_suave['Tamanho MB Suave'] = df_suave.groupby('Base')['Tamanho (MB)'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_suave, x='Data', y='Tamanho MB Suave', hue='Base', ax=ax1, marker='o')
ax1.set_xlabel("Data")
ax1.set_ylabel("Tamanho (MB)")
ax1.set_title("Tamanho com Média Móvel")
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.legend()
st.pyplot(fig1)

# === Projeção ARIMA ===
st.subheader("🔮 Projeção ARIMA para os Próximos 90 Dias")

for base in bases_selecionadas:
    df_base = df_filtrado[df_filtrado['Base'] == base].copy()
    df_base = df_base.sort_values('Data')
    serie = df_base['Tamanho (MB)'].values

    # Ajusta ordem do ARIMA (pode ser ajustado conforme necessidade)
    ordem_arima = (1, 1, 1)
    try:
        modelo = ARIMA(serie, order=ordem_arima)
        modelo_fit = modelo.fit()
        previsoes = modelo_fit.forecast(steps=90)
        datas_futuras = pd.date_range(df_base['Data'].max() + timedelta(days=1), periods=90)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_base, x='Data', y='Tamanho (MB)', ax=ax2, marker='o', label='Histórico')
        ax2.plot(datas_futuras, previsoes, linestyle='--', color='gray', label='Projeção ARIMA')
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Tamanho projetado (MB)")
        ax2.set_title(f"Projeção ARIMA nos Próximos 90 Dias - {base}")
        ax2.grid(True, linestyle='--', linewidth=0.5)
        ax2.legend()
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Não foi possível gerar ARIMA para a base {base}: {e}")

# === Crescimento percentual ===
st.subheader("📉 Crescimento Percentual (%)")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_filtrado, x='Data', y='Crescimento (%)', hue='Base', ax=ax3, marker='o', palette='Set2')
ax3.set_xlabel("Data")
ax3.set_ylabel("Crescimento (%)")
ax3.set_title("Variação Percentual por Base")
ax3.grid(True, linestyle='--', linewidth=0.5)
st.pyplot(fig3)

# === Tabela e download ===
st.subheader("📋 Tabela de Dados Filtrados")
st.dataframe(df_filtrado)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_filtrado.to_excel(writer, index=False, sheet_name='Dados Filtrados')
buffer.seek(0)
st.download_button(
    label="⬇️ Baixar Excel dos dados filtrados",
    data=buffer,
    file_name='dados_filtrados.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Filtrar último mês disponível
ultimo_mes = df['Data'].max().month
ultimo_ano = df['Data'].max().year
df_ultimo_mes = df[(pd.to_datetime(df['Data']).dt.month == ultimo_mes) & (pd.to_datetime(df['Data']).dt.year == ultimo_ano)]

# Gráfico: Top 10 bases do servidor 5
df_s5 = df_ultimo_mes[df_ultimo_mes['Servidor'] == 's5']
top10_s5 = df_s5.groupby('Base')['Tamanho (MB)'].sum().nlargest(10).reset_index()

st.subheader("🏆 Top 10 Bases - Servidor 5 (Último mês)")
fig_s5, ax_s5 = plt.subplots(figsize=(10, 4))
sns.barplot(data=top10_s5, x='Base', y='Tamanho (MB)', color='gold', ax=ax_s5)
ax_s5.set_title(f"Top 10 Bases - Servidor 5 ({ultimo_mes:02d}/{ultimo_ano})")
ax_s5.set_xlabel("Base")
ax_s5.set_ylabel("Tamanho (MB)")
ax_s5.tick_params(axis='x', rotation=45)
st.pyplot(fig_s5)

# Gráfico: Top 10 bases do servidor 6
df_s6 = df_ultimo_mes[df_ultimo_mes['Servidor'] == 's6']
top10_s6 = df_s6.groupby('Base')['Tamanho (MB)'].sum().nlargest(10).reset_index()

st.subheader("🏆 Top 10 Bases - Servidor 6 (Último mês)")
fig_s6, ax_s6 = plt.subplots(figsize=(10, 4))
sns.barplot(data=top10_s6, x='Base', y='Tamanho (MB)', color='deepskyblue', ax=ax_s6)
ax_s6.set_title(f"Top 10 Bases - Servidor 6 ({ultimo_mes:02d}/{ultimo_ano})")
ax_s6.set_xlabel("Base")
ax_s6.set_ylabel("Tamanho (MB)")
ax_s6.tick_params(axis='x', rotation=45)
st.pyplot(fig_s6)

# Gráfico: Total do servidor 5 no último mês
total_s5 = df_s5['Tamanho (MB)'].sum()
st.subheader("📦 Total de Dados - Servidor 5 (Último mês)")
fig_total_s5, ax_total_s5 = plt.subplots(figsize=(4, 4))
ax_total_s5.bar(['Servidor 5'], [total_s5], color='gold')
ax_total_s5.set_ylabel("Tamanho Total (MB)")
ax_total_s5.set_title(f"Total Servidor 5 ({ultimo_mes:02d}/{ultimo_ano})")
st.pyplot(fig_total_s5)

# Gráfico: Total do servidor 6 no último mês
total_s6 = df_s6['Tamanho (MB)'].sum()
st.subheader("📦 Total de Dados - Servidor 6 (Último mês)")
fig_total_s6, ax_total_s6 = plt.subplots(figsize=(4, 4))
ax_total_s6.bar(['Servidor 6'], [total_s6], color='deepskyblue')
ax_total_s6.set_ylabel("Tamanho Total (MB)")
ax_total_s6.set_title(f"Total Servidor 6 ({ultimo_mes:02d}/{ultimo_ano})")
st.pyplot(fig_total_s6)
