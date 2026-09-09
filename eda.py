# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pygam import ExpectileGAM
# %%

df=pd.read_csv('sales_data_analysis.csv')
# %%
print(f'Quantidade de colunas - {str(len(list(df.columns)))}\n{50*"="}')
print(df.dtypes)
print(f'\n{50*"="}\n{df.count()}')
df.head(10)

#%%
df['Manager'].value_counts()
#%%
df['Manager'] = df['Manager'].str.strip().str.replace(r'\s+', ' ', regex=True) 
df['Manager'].value_counts()

#%%
df.dtypes
df['Quantity']=df['Quantity'].astype(int)
df['Revenue']=df['Price']*df['Quantity']
# %%

df.head(20)

#%%
# Criando uma List Comprehension para filtrar colunas categoricas
target_cols = [
    col
    for col, dtype in df.dtypes.items()
    if dtype in ['str'] and df[col].nunique() <= 10
]

# Configurar as grades de visualização dos subplots
n_cols = 2
n_rows = int(np.ceil(len(target_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten() if len(target_cols) > 1 else [axes]

sns.set_theme(style='whitegrid')

# Para cada coluna alvo, gera um boxplot
for i, col in enumerate(target_cols):
    ax = axes[i]

    # Ordena pelas categorias mais frequentes dentro da coluna
    order = df[col].value_counts().index

    # Cria o gráfico de contagem
    sns.countplot(
        data=df,
        y=col,
        order=order,
        #palette='Blues_r',
        ax=ax,
    )

    ax.set_title(f'Distribuição: {col}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Contagem')
    ax.set_ylabel('')

    # Adiciona rótulos com a contagem exata no final de cada barra
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10)

# Esconde eixos sobrando caso o número de colunas seja ímpar
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#%%
df[['Product','Price']].value_counts().sort_values()
# Correção da remoção de linhas com preços corrompidos (29.05 e 25.50 estavam na coluna Price)
df.drop(df.loc[(df['Product']=='Chicken Sandwiches') & (df['Price'].isin([29.05, 29.90]))].index, inplace=True)
df.drop(df.loc[(df['Product']=='Fries') & (df['Price'].isin([25.50, 24.50]))].index, inplace=True)

# %%
def add_price_variation(
    df,
    price_col='Price',
    quantity_col='Quantity',
    city_col='City',
    max_variation=0.25, # Variação máxima de 25%
    elasticity=-1.3, # Elasticidade-preço da demanda
    city_multipliers=None, # Multiplicadores de preço por cidade
    outlier_prob=0.05, # Probabilidade de ocorrência de outliers
    random_state=42 # Seed para reprodutibilidade
):
    """
    Simula uma variação realista de preços e demanda de mercado:
    
    1. Variação por Cidade (City): Reflete custo de vida e poder de compra regional.
       Cidades mais caras (London, Paris) praticam preços ligeiramente maiores,
       enquanto cidades como Madri e Lisboa praticam preços menores. Somado ao
       ruído local diário, a variação chega a até +-25% (max_variation).
       
    2. Elasticidade-Preço da Demanda (Lei da Demanda):
       Q = Q_base * (P / P_base) ** elasticity * ruído_demanda
       - Preços mais baixos geram maior volume vendido (mais casos vendidos).
       - Preços mais altos retraem a demanda (menos casos vendidos).
       
    3. Outliers da Vida Real:
       - Preço: liquidações relâmpago profundas (-30% a -45%) ou cobranças de eventos/aeroporto (+30% a +45%).
       - Demanda: pedidos volumosos em lote/catering (bulk) ou rupturas operacionais de estoque (stockouts).
       
    4. Atualização de Receita: df['Revenue'] = df['Price'] * df['Quantity'].
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Preço e quantidade base por produto (usando mediana para robustez contra ruídos)
    base_prices = df.groupby('Product')[price_col].transform('median')
    base_quantities = df.groupby('Product')[quantity_col].transform('median') if quantity_col in df.columns else None

    # Efeito regional por Cidade
    if city_multipliers is None:
        city_multipliers = {
            'London': 0.10,
            'Paris': 0.07,
            'Berlin': 0.00,
            'Madrid': -0.06,
            'Lisbon': -0.09
        }
    
    city_effect = df[city_col].map(city_multipliers).fillna(0.0) if city_col in df.columns else 0.0
    
    # Flutuação local/diária distribuída normalmente, delimitada em +-max_variation (25%)
    max_city_effect = np.abs(list(city_multipliers.values())).max() if city_multipliers else 0.0
    remaining_var = max(0.05, max_variation - max_city_effect)
    local_noise = rng.normal(0, remaining_var / 2.0, size=len(df))
    
    total_pct_var = np.clip(city_effect + local_noise, -max_variation, max_variation)
    
    # 1. Outliers de Preço (promoções extremas ou sobrepreço de eventos)
    is_price_outlier = rng.random(size=len(df)) < (outlier_prob / 2.0)
    outlier_direction = rng.choice([-1, 1], size=len(df))
    extreme_price_var = outlier_direction * rng.uniform(0.30, 0.45, size=len(df))
    total_pct_var = np.where(is_price_outlier, extreme_price_var, total_pct_var)
    
    # Atualiza coluna de Preço
    df[price_col] = (base_prices * (1 + total_pct_var)).round(2)
    df[price_col] = np.maximum(df[price_col], 0.50) # salvaguarda mínima de valor
    
    # 2. Elasticidade da Demanda (atualização da Quantidade vendida)
    if base_quantities is not None:
        price_ratio = df[price_col] / base_prices
        
        # Suporte a elasticidade específica por produto ou genérica
        if isinstance(elasticity, dict):
            prod_elasticity = df['Product'].map(elasticity).fillna(-1.2)
        else:
            prod_elasticity = elasticity
            
        # Ruído estocástico diário na demanda (clima, dia da semana, fluxo de clientes)
        daily_demand_noise = rng.normal(1.0, 0.06, size=len(df))
        simulated_qty = base_quantities * (price_ratio ** prod_elasticity) * daily_demand_noise
        
        # 3. Outliers de Demanda (cenários atípicos da vida real)
        is_demand_outlier = rng.random(size=len(df)) < (outlier_prob / 2.0)
        outlier_type = rng.choice(['bulk', 'stockout'], size=len(df), p=[0.6, 0.4])
        
        bulk_factor = rng.uniform(1.8, 2.8, size=len(df))       # Pedidos para eventos/empresas
        stockout_factor = rng.uniform(0.15, 0.35, size=len(df)) # Falta de suprimento/estoque
        
        simulated_qty = np.where(
            is_demand_outlier & (outlier_type == 'bulk'),
            simulated_qty * bulk_factor,
            simulated_qty
        )
        simulated_qty = np.where(
            is_demand_outlier & (outlier_type == 'stockout'),
            simulated_qty * stockout_factor,
            simulated_qty
        )
        
        df[quantity_col] = np.round(np.maximum(simulated_qty, 1)).astype(int)
    
    # 4. Atualização da Receita
    if 'Revenue' in df.columns or quantity_col in df.columns:
        df['Revenue'] = (df[price_col] * df[quantity_col]).round(2)
        
    return df

# Aplica variação realista de preços (até 25%), efeito por cidade, elasticidade e outliers
df = add_price_variation(
    df,
    price_col='Price',
    quantity_col='Quantity',
    city_col='City',
    max_variation=0.25,
    elasticity=-1.3,
    outlier_prob=0.05,
    random_state=42
)

# %%
df[['Product','Price']].loc[df['Product']=='Chicken Sandwiches'].sort_values('Price', ascending=True)

#%%

df.loc[(df['Product']=='Fries')].sort_values('Quantity', ascending=False).head(20)

# %%
# Criando o boxplot se baseando na categoria
sns.violinplot(data=df, 
            x='Category', 
            y='Price',
            palette='Set2')

plt.title('Distribuição do Preço por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Preço ($)')
plt.show()

sns.violinplot(data=df, 
            x='Manufacturer', 
            y='Price',
            palette='Set2')

plt.title('Distribuição do Preço por Fabricante')
plt.xlabel('Manufacturer')
plt.ylabel('Preço ($)')
plt.show()


# %%
import plotly.express as px

fig = px.scatter(
    df, 
    x='Price', 
    y='Quantity', 
    color='Product',
    #hover_data=['Purchase Type', 'City', 'Manager'], # Mostra informações extras ao passar o mouse
    title='Elasticidade de Demanda: Preço vs. Quantidade por Produto',
    labels={'Price': 'Preço ($)', 'Quantity': 'Quantidade Vendida'},
    opacity=0.7,
    template='plotly_white',
    marginal_x='box', # Adiciona boxplots nas margens para mostrar mín, máx e mediana do Preço
    marginal_y='box'  # Adiciona boxplots nas margens para mostrar mín, máx e mediana da Quantidade
)

# Melhorando a aparência dos pontos
fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(type='scatter'))

# Exibindo o gráfico interativo
fig.show()



# %%
# Visualização por Expectile dos preços se baseando na categoria


quantiles=[0.1,0.25,0.5,0.75,0.9]
gam_results={}
x=df['Price']
y=df['Quantity']

for q in quantiles:
    gam= ExpectileGAM(terms='auto',expectile=q)
    gam.fit(x, y)
    gam_results[q] = gam

# %%
sns.scatterplot(x=x,
            y=y,
            hue=df['Product']            
            )

# %%

# Criando uma List Comprehension para filtrar colunas categoricas
target_cols = [
    col
    for col, dtype in df.dtypes.items()
    if dtype in ['str'] and df[col].nunique() <= 10
]

# Configurar as grades de visualização dos subplots
n_cols = 2
n_rows = int(np.ceil(len(target_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten() if len(target_cols) > 1 else [axes]

sns.set_theme(style='whitegrid')

# Para cada coluna alvo, gera um boxplot
for i, col in enumerate(target_cols):
    ax = axes[i]

    # Ordena pelas categorias mais frequentes dentro da coluna
    order = df[col].value_counts().index

    # Cria o gráfico de contagem
    sns.countplot(
        data=df,
        y=col,
        order=order,
        #palette='Blues_r',
        ax=ax,
    )

    ax.set_title(f'Distribuição: {col}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Contagem')
    ax.set_ylabel('')

    # Adiciona rótulos com a contagem exata no final de cada barra
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10)

# Esconde eixos sobrando caso o número de colunas seja ímpar
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# %%
# Criando o boxplot se baseando na categoria
sns.violinplot(data=df, 
            x='Category', 
            y='Price',
            palette='Set2')

plt.title('Distribuição do Preço por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Preço ($)')
plt.show()

sns.violinplot(data=df, 
            x='Manufacturer', 
            y='Price',
            palette='Set2')

plt.title('Distribuição do Preço por Fabricante')
plt.xlabel('Manufacturer')
plt.ylabel('Preço ($)')
plt.show()
# %%
# Visualização por Expectile dos preços se baseando na categoria
quantiles=[0.1,0.25,0.5,0.75,0.9]
gam_results={}
x=df['Price']
y=df['Quantity']

for q in quantiles:
    gam= ExpectileGAM(terms='auto',expectile=q)
    gam.fit(x, y)
    gam_results[q] = gam

# %%
sns.scatterplot(x=x,
            y=y,
            hue=df['City']            
            )
 # %%
