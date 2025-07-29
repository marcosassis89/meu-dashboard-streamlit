import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import io
from datetime import timedelta

# === Carregar dados ===
df = pd.read_excel('../data_raw/saida.xlsx', sheet_name='Crescimento (%)')
df['Data'] = pd.to_datetime(df['Data']).dt.date  # remove hora
df['Tamanho GB'] = df['Tamanho'] / (1024 ** 3)   # bytes → GB

# === Corrigir cálculo de crescimento percentual ===
df = df.sort_values(['Base', 'Data'])  # garante ordenação
def calcular_crescimento_percentual(grupo):
    grupo = grupo.sort_values('Data').copy()
    tamanhos = grupo['Tamanho GB'].values
    crescimento = [0.0]  # primeiro sempre 0%
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
df_suave['Tamanho GB Suave'] = df_suave.groupby('Base')['Tamanho GB'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

df_tend = df_filtrado.copy()
df_tend['Dias'] = (pd.to_datetime(df_tend['Data']) - pd.to_datetime(df_tend['Data'].min())).dt.days
modelo_poly = np.polyfit(df_tend['Dias'], df_tend['Tamanho GB'], 2)
df_tend['Tendência'] = modelo_poly[0]*df_tend['Dias']**2 + modelo_poly[1]*df_tend['Dias'] + modelo_poly[2]

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_suave, x='Data', y='Tamanho GB Suave', hue='Base', ax=ax1, marker='o')
ax1.plot(df_tend['Data'], df_tend['Tendência'], color='black', linestyle='-', label='Tendência (Grau 2)')
ax1.set_xlabel("Data")
ax1.set_ylabel("Tamanho (GB)")
ax1.set_title("Tamanho com Média Móvel e Tendência Polinomial")
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.legend()
st.pyplot(fig1)

# === Projeção linear ===
st.subheader("🔮 Projeção Linear para os Próximos 90 Dias")

df_proj = df_filtrado.copy()
df_proj['Dias'] = (pd.to_datetime(df_proj['Data']) - pd.to_datetime(df_proj['Data'].min())).dt.days
modelo_proj = np.polyfit(df_proj['Dias'], df_proj['Tamanho GB'], 1)

dias_futuros = pd.date_range(df_proj['Data'].max() + timedelta(days=1), periods=90)
dias_futuros_int = (dias_futuros - pd.to_datetime(df_proj['Data'].min())).days
projecoes = modelo_proj[0] * dias_futuros_int + modelo_proj[1]

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_proj, x='Data', y='Tamanho GB', hue='Base', ax=ax2, marker='o', legend=False)
ax2.plot(dias_futuros, projecoes, linestyle='--', label='Projeção Linear', color='gray')
ax2.set_xlabel("Data")
ax2.set_ylabel("Tamanho projetado (GB)")
ax2.set_title("Projeção de Crescimento nos Próximos 90 Dias")
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend()
st.pyplot(fig2)

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

# Criar buffer de bytes
buffer = io.BytesIO()

# Exportar para Excel dentro do buffer
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_filtrado.to_excel(writer, index=False, sheet_name='Dados Filtrados')

# Retornar ao início do buffer
buffer.seek(0)

# Botão de download como Excel
st.download_button(
    label="⬇️ Baixar Excel dos dados filtrados",
    data=buffer,
    file_name='dados_filtrados.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)









