import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.express as px
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

# === Ranking de crescimento ===
def ranking_crescimento(df):
    st.subheader("🚀 Ranking de Crescimento (%)")
    df_agg = df.groupby('Base').agg({
        'Crescimento (%)': 'mean',
        'Tamanho (MB)': lambda x: x.diff().mean()
    }).sort_values('Crescimento (%)', ascending=False).reset_index()
    df_agg = df_agg.rename(columns={
        'Crescimento (%)': 'Crescimento Médio (%)',
        'Tamanho (MB)': 'Crescimento Médio (MB)'
    })
    st.dataframe(df_agg.head(10))

# Chamada da função (fora da definição)
ranking_crescimento(df_filtrado)

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

# === Definir último mês e ano com base na data mais recente ===
ultima_data = pd.to_datetime(df['Data']).max()
ultimo_mes = ultima_data.month
ultimo_ano = ultima_data.year

# === Filtrar último registro por base no último mês ===
df_ultimo_mes = df[
    (pd.to_datetime(df['Data']).dt.month == ultimo_mes) &
    (pd.to_datetime(df['Data']).dt.year == ultimo_ano)
].copy()

df_ultimo_mes_atual = (
    df_ultimo_mes.sort_values('Data')
    .groupby(['Servidor', 'Base'], as_index=False)
    .last()
)

# === Função para gráfico Top 10 por servidor ===
def plot_top10(df_servidor, servidor_nome, cor):
    top10 = df_servidor.nlargest(10, 'Tamanho (MB)')
    st.subheader(f"🏆 Top 10 Bases - Servidor {servidor_nome} ({ultimo_mes:02d}/{ultimo_ano})")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top10, x='Base', y='Tamanho (MB)', color=cor, ax=ax)
    ax.set_title(f"Top 10 Bases - Servidor {servidor_nome} ({ultimo_mes:02d}/{ultimo_ano})")
    ax.set_xlabel("Base")
    ax.set_ylabel("Tamanho (MB)")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    st.pyplot(fig)

# === Gráficos Top 10 por servidor ===
df_s5 = df_ultimo_mes_atual[df_ultimo_mes_atual['Servidor'] == 's5']
df_s6 = df_ultimo_mes_atual[df_ultimo_mes_atual['Servidor'] == 's6']

if not df_s5.empty:
    plot_top10(df_s5, '5', 'gold')
else:
    st.info("ℹ️ Nenhum dado disponível para o Servidor 5 no último mês.")

if not df_s6.empty:
    plot_top10(df_s6, '6', 'deepskyblue')
else:
    st.info("ℹ️ Nenhum dado disponível para o Servidor 6 no último mês.")

# === Gráfico unificado: Total por servidor ===
total_s5 = df_s5['Tamanho (MB)'].sum()
total_s6 = df_s6['Tamanho (MB)'].sum()

st.subheader(f"📦 Total de Dados - Servidores 5 e 6 ({ultimo_mes:02d}/{ultimo_ano})")
fig_total, ax_total = plt.subplots(figsize=(6, 4))
ax_total.bar(['Servidor 5', 'Servidor 6'], [total_s5, total_s6], color=['gold', 'deepskyblue'])
ax_total.set_ylabel("Tamanho Total (MB)")
ax_total.set_title(f"Total Servidores 5 e 6 ({ultimo_mes:02d}/{ultimo_ano})")
ax_total.grid(True, axis='y', linestyle='--', linewidth=0.5)
ax_total.grid(True, axis='x', linestyle='--', linewidth=0.5)
st.pyplot(fig_total)

# === Crescimento por base com seleção de servidor e filtro por data ===
st.subheader("📊 Crescimento por Base por Servidor e Período")

# Garantir que a coluna 'Diferença (MB)' esteja em formato numérico
df['Diferença (MB)'] = pd.to_numeric(df['Diferença (MB)'], errors='coerce')

# Garantir que a coluna de data esteja em formato datetime
df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

# Selectbox para escolha do servidor
servidor_selecionado = st.selectbox("Selecione o servidor:", options=['s5', 's6'])

# Filtro de data
data_min = df['Data'].min()
data_max = df['Data'].max()
data_inicio, data_fim = st.date_input("Selecione o intervalo de datas:",
                                      value=(data_min, data_max),
                                      min_value=data_min,
                                      max_value=data_max)

# Filtrar dados conforme seleção
df_filtrado = df[
    (df['Servidor'] == servidor_selecionado) &
    (df['Data'] >= pd.to_datetime(data_inicio)) &
    (df['Data'] <= pd.to_datetime(data_fim))
].copy()

# Agrupar por base e calcular crescimento real (final - inicial)
crescimento_por_base = []
for base in df_filtrado['Base'].unique():
    df_base = df_filtrado[df_filtrado['Base'] == base].sort_values('Data')
    if len(df_base) < 2:
        continue
    tamanho_inicial = df_base['Tamanho (MB)'].iloc[0]
    tamanho_final = df_base['Tamanho (MB)'].iloc[-1]
    crescimento_mb = tamanho_final - tamanho_inicial
    crescimento_por_base.append({
        'Base': base,
        'Tamanho Inicial (MB)': tamanho_inicial,
        'Tamanho Final (MB)': tamanho_final,
        'Crescimento (MB)': crescimento_mb
    })

crescimento_por_base_df = pd.DataFrame(crescimento_por_base).sort_values('Crescimento (MB)', ascending=False)

# Mostrar tabela
st.dataframe(crescimento_por_base_df)

# Slider para limitar número de bases no gráfico
top_n = st.slider("Número de bases a exibir no gráfico:", min_value=5, max_value=30, value=15)
dados_grafico = crescimento_por_base_df.head(top_n)

# Gráfico horizontal para melhor legibilidade
fig, ax = plt.subplots(figsize=(10, len(dados_grafico) * 0.4))
palette = sns.color_palette("Purples", len(dados_grafico)) if servidor_selecionado == 's5' else sns.color_palette("magma", len(dados_grafico))
sns.barplot(data=dados_grafico, y='Base', x='Crescimento (MB)', palette=palette, ax=ax)

# Títulos e rótulos
ax.set_title(f"Crescimento por Base - Servidor {servidor_selecionado} ({data_inicio} a {data_fim})")
ax.set_xlabel("Crescimento (MB)")
ax.set_ylabel("Base")
ax.grid(True, axis='x', linestyle='--', linewidth=0.5)

# Exibir gráfico
st.pyplot(fig)

# Mostrar crescimento total
crescimento_total = crescimento_por_base_df['Crescimento (MB)'].sum()
st.markdown(f"**📦 Crescimento total do servidor {servidor_selecionado} no período:** `{crescimento_total:.2f} MB`")

# === Evolução do Tamanho por Base no Período Selecionado ===
st.subheader("📈 Evolução do Tamanho por Base no Período Selecionado")

# Limite de alerta para crescimento em MB
limite_alerta_mb = st.slider("Defina o limite de alerta para crescimento (MB):", min_value=1.0, max_value=100.0, value=20.0)

# Calcular crescimento absoluto e percentual por base
df_evolucao = df_filtrado.sort_values(['Base', 'Data']).copy()
bases = df_evolucao['Base'].unique()

dados_crescimento = []
for base in bases:
    df_base = df_evolucao[df_evolucao['Base'] == base]
    if len(df_base) < 2:
        continue
    tamanho_inicial = df_base['Tamanho (MB)'].iloc[0]
    tamanho_final = df_base['Tamanho (MB)'].iloc[-1]
    crescimento_mb = tamanho_final - tamanho_inicial
    crescimento_pct = ((tamanho_final - tamanho_inicial) / tamanho_inicial) * 100 if tamanho_inicial != 0 else 0
    dados_crescimento.append({
        'Base': base,
        'Tamanho Inicial (MB)': tamanho_inicial,
        'Tamanho Final (MB)': tamanho_final,
        'Crescimento (MB)': crescimento_mb,
        'Crescimento (%)': crescimento_pct
    })

df_crescimento = pd.DataFrame(dados_crescimento)
df_crescimento = df_crescimento.sort_values('Crescimento (MB)', ascending=False)

# Destacar bases com crescimento acima do limite
bases_alerta = df_crescimento[df_crescimento['Crescimento (MB)'] > limite_alerta_mb]
if not bases_alerta.empty:
    st.warning(f"🚨 {len(bases_alerta)} base(s) tiveram crescimento acima de {limite_alerta_mb:.2f} MB no período selecionado.")
    st.dataframe(bases_alerta.style.format({
        'Tamanho Inicial (MB)': '{:.2f}',
        'Tamanho Final (MB)': '{:.2f}',
        'Crescimento (MB)': '{:.2f}',
        'Crescimento (%)': '{:.2f}%'
    }))
else:
    st.info("✅ Nenhuma base ultrapassou o limite de crescimento definido.")

# Mostrar tabela completa
st.markdown("### 📋 Crescimento por Base no Período")
st.dataframe(df_crescimento.style.format({
    'Tamanho Inicial (MB)': '{:.2f}',
    'Tamanho Final (MB)': '{:.2f}',
    'Crescimento (MB)': '{:.2f}',
    'Crescimento (%)': '{:.2f}%'
}))

# Gráfico de linha por base
st.markdown("### 📈 Evolução Interativa do Tamanho por Base")

fig_plotly = px.line(
    df_evolucao,
    x='Data',
    y='Tamanho (MB)',
    color='Base',
    markers=True,
    title=f"Evolução do Tamanho por Base - Servidor {servidor_selecionado}",
    labels={'Data': 'Data', 'Tamanho (MB)': 'Tamanho (MB)', 'Base': 'Base'}
)
fig_plotly.update_layout(
    legend_title_text='Base',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    height=600
)
st.plotly_chart(fig_plotly, use_container_width=True)
