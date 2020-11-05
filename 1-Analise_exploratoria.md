https://www.datacamp.com/community/tutorials/markdown-in-jupyter-notebook
https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html


# Análise Exploratória dos Dados
Este script tem como objetivo explorar a base de dados:
- identificar os atributos disponíveis
- Tipos de dados
- Comportamento dos dados
- Valores nulos e/ou discrepantes (outlier)
- e fazer o tratamento dos dados necessário


1. [Seleção dos Dados](#selecao)
2. [Exploração dos Dados](#exploracao)
3. [Tratamento dos Dados](#tratamento)
4. [Análise Exploratória II](#exploracao_y)
5. [Conclusão](#conclusao)
6. [Próximas etapas](#proxima)

<font color='red'> Este script seria utilizado apenas pela equipe técnica </font>


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

```


```python
## função para calcular limite superior e inferior dos dados
def calc_limit_boxplot(data):
    q1 = data.quantile(.25)
    q3 = data.quantile(.75)
    
    sup = q3 + 1.5 * (q3-q1)
    inf = q1 - 1.5 * (q3-q1)
    
    return sup, inf
```


```python
cores_classe = [sns.color_palette("hls", 5)[0],sns.color_palette("hls", 5)[2]]
sns.set_palette(cores_classe)
```

## <a id = "selecao"> 1. Seleção dos dados </a>



```python
dados = pd.read_csv('data/campaigns.csv')
```

- Identificando as atributos(nomes) e tipos de dados armazenados


```python
dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40690 entries, 0 to 40689
    Data columns (total 18 columns):
    Unnamed: 0    40690 non-null int64
    age           40690 non-null int64
    job           40690 non-null object
    marital       40690 non-null object
    education     40690 non-null object
    default       40690 non-null object
    balance       40690 non-null int64
    housing       40690 non-null object
    loan          40690 non-null object
    contact       40690 non-null object
    day           40690 non-null int64
    month         40690 non-null object
    duration      40690 non-null int64
    campaign      40690 non-null int64
    pdays         40690 non-null int64
    previous      40690 non-null int64
    poutcome      40690 non-null object
    y             40690 non-null object
    dtypes: int64(8), object(10)
    memory usage: 5.6+ MB
    

- É possível observar que há muitos atributos do tipo qualitativo

- É possível notar que não há nenhum valor faltante em nenhuma das colunas da base

<font color='red'> Excluindo atributos irrelevante </font>


```python
## excluindo colunas
dados.drop(columns=['Unnamed: 0'], inplace=True)
```

<font color='red'> Estatística descritiva dos campos numéricos</font>



```python
dados.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>balance</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
      <td>40690.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>40.905407</td>
      <td>1359.697518</td>
      <td>15.808405</td>
      <td>258.243844</td>
      <td>2.764586</td>
      <td>40.059867</td>
      <td>0.579405</td>
    </tr>
    <tr>
      <td>std</td>
      <td>10.604908</td>
      <td>3034.248783</td>
      <td>8.318281</td>
      <td>257.577068</td>
      <td>3.110158</td>
      <td>100.078281</td>
      <td>2.350664</td>
    </tr>
    <tr>
      <td>min</td>
      <td>18.000000</td>
      <td>-8019.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>33.000000</td>
      <td>74.000000</td>
      <td>8.000000</td>
      <td>103.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>39.000000</td>
      <td>451.000000</td>
      <td>16.000000</td>
      <td>180.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>48.000000</td>
      <td>1423.000000</td>
      <td>21.000000</td>
      <td>319.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>95.000000</td>
      <td>102127.000000</td>
      <td>31.000000</td>
      <td>4918.000000</td>
      <td>63.000000</td>
      <td>871.000000</td>
      <td>275.000000</td>
    </tr>
  </tbody>
</table>
</div>



- podemos observar que os dados estão bem dispersos, não há uma distribuião padrão e estão com escalas bem distintas


## <a id = "exploracao">2. Exploração dos Dados </a>

A partir desta seção serão feitas algumas explorações da base de dados para saber como os dados se comportam, estão distribuídos dentre outras. 

<font color='red'> **Exploração das variáveis quantitativas** </font>


```python
dados.hist(figsize=(15,15), color = 'blue')
plt.show()
    
```


![png](output_15_0.png)


Com base na visualização acima podemos observar algumas pontos, como:
- `age`: boa parte dos clientes tem entre 25 e 50 anos.
- `balance`: boa parte dos cliente tem um saldo médio negativo
- `campaign`: foram feitas até 3 ligações para os cliente sobre esta campanha
- `day`: não há um padrão de comportamento para o dia da última ligação, é bem aleatório o comportamente.
- `duration`: as ligações duram majoritariomente até 500 segundos
- `pday`: maioria da vezes a empresa retornou para o cliente 100 dias após o último contato
- `previous`: majoritariamente dos cliente não foram contatados em campanhas anteriores.
   

 
<font color='red'> **Exploração das variáveis qualitativas** </font>


```python
c,r = 0,0
fig, axes = plt.subplots(3,3, figsize = (20,12))

for dtype in zip(dados.dtypes.index,dados.dtypes):
    
    if dtype[0] not in ['y'] and dtype[1] == object:
        #print(dtype[0], r,c)#print(dtype[0])
        #chart = 
        #sns.distplot(d, kde=False, color="b", ax=axes[0, 0])
        order = dados[dtype[0]].value_counts().index

        #chart = sns.countplot(x = 'job', hue = 'y', data=dados_selec, order = order)
        chart = sns.countplot(x = dtype[0], data=dados, ax = axes[r,c], color = 'blue', order = order)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', size = 11)
        #chart.set_yticklabels(chart.get_yticks(), size = 5)
        #plt.title('Frequência do atributo '+dtype[0])
        axes[r,c].set_title(dtype[0])
        fig.tight_layout()
        #sns.despine(left=True)
        if c == 2:
            r += 1
            c = 0
        else: c += 1
        
            
plt.show()
```


![png](output_17_0.png)


Com base nessas visualizações:
- `job`: há 12 tipos de cargos na base de clientes, e boa parte dos clientes trabalham como 'blue-collar'(trabalho manual), 'management' e 'technician'(área de tecnológia);
- `marital`: majoritariamente os clientes dessa empresa são casados
- `education`: há 4 nível de escolaridade na base de dados, a maioria tem o nível 'secondary'
- `default`: majoritariamente os cliente não tem crédito
- `housing`: há uma divisão equilibrada para os cliente que tem empréstimo(financiamento) de imóveis e os que não tem
- `loan`: boa parte dos clientes não tem empréstimo
- `contact`: há 3 formas de contato com o cliente, mas boa parte feita por celular
- `month`: a empresa entra em contato com os cliente principalmente no mês de maio e pouco contato para o final do ano (outubro-dezembro)
- `poutcome`:  como identificado anteriormente, boa parte dos dados foi primeiro contato, então para campanhas anteriores maioria não terá essa informação, seguida por 'failure' (cliente não aderiu a campanha anterior)



```python
qtd_y = dados['y'].value_counts()
qtd_y.plot.pie(title = 'Distribuição da Classe (base original)',
                      autopct='%.1f%%', startangle=90)

plt.show()
```


![png](output_19_0.png)


- É possível observar que há uma grande diferença entre as duas classes, isso pode ser um problema para o modelo de machine learning, mas será tratado mais a frente

### Conclusão Parcial
- Análise exploratória inicial foi apenas para conhecer a base de dados, identificar quais atributos e quais tipos de dados;
- também foi possível observar que alguns atributos tem comportamente bastante desbalanceado, por exemplo, `defaul`, `loan`. Porém, esse desbalanceamento não interfira nas análises.
- Por fim, identificamos que há um desbalanceamento considerável no atributo `y`. Este desbalanceamento pode interferir na elaboração do modelo para classificação. Mas será tratado mais a frente.

## <a id = "tratamento">3. Tratamento dos Dados </a>

Nessa seção serão realizadas algumas explorações mais afundo sobre os dados e alguns tratamentos (limpeza, transformação etc) dos dados.

<font color='red'> Transformando o atributo `month` para quantitativo (Criando nova coluna de mês com dado quantitativo) </font>



```python
## tratamento campo mês(para melhorar a visualizaão)
replace_month = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
dados['month_new'] = dados.replace({'month':replace_month},inplace=False)['month']

```

<font color='red'> convertendo variáveis qualitativas em quantitativas (Esse procedimento foi realizado para permitir fazer algumas análise estatísticas mais a frente) </font>



```python
le = LabelEncoder()
for dtype in zip(dados.dtypes.index,dados.dtypes):
    if dtype[0] not in ['month'] and dtype[1] == object:
        print(dtype[0])
        dados[dtype[0]+'_new'] = le.fit_transform(dados[dtype[0]])

```

    job
    marital
    education
    default
    housing
    loan
    contact
    poutcome
    y
    


```python
## verificando novas colunas criadas
dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40690 entries, 0 to 40689
    Data columns (total 27 columns):
    age              40690 non-null int64
    job              40690 non-null object
    marital          40690 non-null object
    education        40690 non-null object
    default          40690 non-null object
    balance          40690 non-null int64
    housing          40690 non-null object
    loan             40690 non-null object
    contact          40690 non-null object
    day              40690 non-null int64
    month            40690 non-null object
    duration         40690 non-null int64
    campaign         40690 non-null int64
    pdays            40690 non-null int64
    previous         40690 non-null int64
    poutcome         40690 non-null object
    y                40690 non-null object
    month_new        40690 non-null int64
    job_new          40690 non-null int32
    marital_new      40690 non-null int32
    education_new    40690 non-null int32
    default_new      40690 non-null int32
    housing_new      40690 non-null int32
    loan_new         40690 non-null int32
    contact_new      40690 non-null int32
    poutcome_new     40690 non-null int32
    y_new            40690 non-null int32
    dtypes: int32(9), int64(8), object(10)
    memory usage: 7.0+ MB
    


```python
colunas_numericas = ['age','balance','campaign','day','duration','pdays','previous','y']
```

<font color='red'>Análise do comportamento(distribuição) dos dados numéricos separados pela classe (Y)</font>


```python
plt.figure(figsize=(10,5))
sns.pairplot(dados[colunas_numericas], hue= 'y')
plt.show()
```


    <Figure size 720x360 with 0 Axes>



![png](output_29_1.png)


- Podemos observar que há nenhuma relação entre as variáveis
- É notado alguns que alguns atributos possuem outlier, como, `previus` e `campaign`

<font color='red'> **Limpeza dos Outlier** </font>

eliminando outlier da coluna `previous` e `campaign`, utilizando separatiz (quartil)


```python
print('base original: ', len(dados))
dados_clean = dados[dados.previous < dados.previous.quantile(.999)]
dados_clean = dados_clean[dados_clean.campaign < dados_clean.campaign.quantile(.99)]

print('base limpa: ', len(dados_clean))
```

    base original:  40690
    base limpa:  40234
    

<font color='red'> Novo plot após eliminar outlier </font>



```python
plt.figure(figsize=(10,5))
sns.pairplot(dados_clean[colunas_numericas], hue= 'y')
plt.show()
```


    <Figure size 720x360 with 0 Axes>



![png](output_33_1.png)


- Após eliminar os outlier o comportamento dos dados melhorou um pouco mas ainda há alguns valores um tanto pouco discrepantes, por exemplo, em `duration`, `pdays`.


<font color='red'> Analisando Dados com Boxplot</font>



```python
## escala log para melhorar a visualização
dados_clean['balance_log'] = np.log(dados_clean.balance)
dados_clean['duration_log'] = np.log(dados_clean.duration)

```

    C:\DataScience\ProgramData\Anaconda3\lib\site-packages\pandas\core\series.py:853: RuntimeWarning: divide by zero encountered in log
      result = getattr(ufunc, method)(*inputs, **kwargs)
    C:\DataScience\ProgramData\Anaconda3\lib\site-packages\pandas\core\series.py:853: RuntimeWarning: invalid value encountered in log
      result = getattr(ufunc, method)(*inputs, **kwargs)
    


```python

```


```python
from matplotlib.gridspec import GridSpec
gs = GridSpec(3,3,bottom=200,top=201,left=200,right=201,wspace=.5,hspace=.5) ## linas x colunas
plt.figure(figsize=(10,10))

for i, feature_name in enumerate(colunas_numericas[:-1] + ['balance_log','duration_log']):
    #print(i,feature_name)
    ax = plt.subplot(gs[i])
    #sns.boxplot(dados[feature_name], hue='y')
    sns.boxplot(y=feature_name, data=dados_clean)  
    #plt.legend()
    ax.set_xlabel('')
    ax.set_title(str(feature_name))
```


![png](output_37_0.png)


- é possível identificar muitos outlier em quase todas as colunas
- podemos observar que a coluna `previous` e `pday` tem uma distribuião de valores muito ruim, mesmo eliminando os outliers anteriormente.

<font color='red'> Remoção dos outlier </font>

Nessa remoção dos oulier será utilizado o calculo do limite superior e inferior das coluna `age`,`campaign`,`balance`,`duration`.


```python
print('base original: ', len(dados))
dados_clean = dados_clean[dados_clean.age <= calc_limit_boxplot(dados_clean['age'])[0] ]

dados_clean = dados_clean[dados_clean.campaign <= calc_limit_boxplot(dados_clean['campaign'])[0] ]

dados_clean = dados_clean[(dados_clean.balance <= calc_limit_boxplot(dados_clean['balance'])[0]) & 
                          (dados_clean.balance >= calc_limit_boxplot(dados_clean['balance'])[1])]

dados_clean = dados_clean[(dados_clean.duration <= calc_limit_boxplot(dados_clean['duration'])[0]) &
                         (dados_clean.duration >= calc_limit_boxplot(dados_clean['duration'])[1])]

print('base limpa: ', len(dados_clean))
```

    base original:  40690
    base limpa:  31140
    

<font color='red'> Novo plot (boxplot) após eliminar outlier </font>



```python
from matplotlib.gridspec import GridSpec
gs = GridSpec(3,3,bottom=200,top=201,left=200,right=201,wspace=.5,hspace=.5) ## linas x colunas
plt.figure(figsize=(10,10))

for i, feature_name in enumerate(colunas_numericas[:-1]):
    #print(i,feature_name)
    ax = plt.subplot(gs[i])
    #sns.boxplot(dados[feature_name], hue='y')
    sns.boxplot(y=feature_name, data=dados_clean)  
    #plt.legend()
    ax.set_xlabel('')
    ax.set_title(str(feature_name))
```


![png](output_41_0.png)


- conseguimos normalizar os dados, mas ainda há algumas colunas com muitos outliers
- mesmo com essa limpeza as colunas `pdays` e `previous` não alteraram nada, mais um indicativo que elas podem ser irrelevantes.
- esse procedimento diminuiu o tamanho da base de dados (por isso que a eliminação dos registros não é tão recomendado a ser feito de primeira)

<font color='red'> Correção dos outlier </font>

Como ainda há outlier algumas coluns, vamos tentar corrigir estes valores utilizando o valor do limite superior e inferior dos dados. (pode-se utilizar a média ou mediana, dependendo de como está a distribuição dos dados)


```python
for c in ['balance', 'duration']:
    print('{} - media: {:.2f} - mediana: {} - Limites: {}'.format(c, dados_clean[c].mean(), dados_clean[c].median(),
                                                                  calc_limit_boxplot(dados_clean[c])))
    

```

    balance - media: 630.00 - mediana: 344.0 - Limites: (2346.0, -1334.0)
    duration - media: 206.55 - mediana: 171.0 - Limites: (547.0, -165.0)
    


```python
## CORREÇÃO DE OUTLIER: age, balance, campaign, duration

## outlier superior
dados_clean['balance_new'] = np.where(dados_clean.balance >= calc_limit_boxplot(dados_clean['balance'])[0],
                                         calc_limit_boxplot(dados_clean['balance'])[0],
                                          #dados_clean.balance.median(),
                                         dados_clean.balance)
## outlier inferior
dados_clean['balance_new'] = np.where(dados_clean.balance <= calc_limit_boxplot(dados_clean['balance'])[1],
                                         calc_limit_boxplot(dados_clean['balance'])[1],
                                         #dados_clean.balance.median(),
                                      dados_clean.balance)

dados_clean['duration_new'] = np.where(dados_clean.duration >= calc_limit_boxplot(dados_clean['duration'])[0],
                                         calc_limit_boxplot(dados_clean['duration'])[0],
                                         dados_clean.duration)
```


```python
from matplotlib.gridspec import GridSpec
gs = GridSpec(3,3,bottom=200,top=201,left=200,right=201,wspace=.5,hspace=.5) ## linas x colunas
plt.figure(figsize=(10,10))

for i, feature_name in enumerate(colunas_numericas[:-1] + ['balance_new','duration_new']):
    #print(i,feature_name)
    ax = plt.subplot(gs[i])
    #sns.boxplot(dados[feature_name], hue='y')
    sns.boxplot(y=feature_name, data=dados_clean)  
    #plt.legend()
    ax.set_xlabel('')
    ax.set_title(str(feature_name))
```


![png](output_45_0.png)


- podemos observar que os valores discrepantes foram tratados, com execeção da coluna `balance`, que mesmo assim ainda continua com outlier mesmo após a correção.
- para este caso, deveríamos nos aprofundar nas análise estatística dessa coluna em específico.



```python
dados_clean.describe().round()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>balance</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>month_new</th>
      <th>job_new</th>
      <th>marital_new</th>
      <th>...</th>
      <th>default_new</th>
      <th>housing_new</th>
      <th>loan_new</th>
      <th>contact_new</th>
      <th>poutcome_new</th>
      <th>y_new</th>
      <th>balance_log</th>
      <th>duration_log</th>
      <th>balance_new</th>
      <th>duration_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>...</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>28245.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
      <td>31140.0</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>40.0</td>
      <td>630.0</td>
      <td>15.0</td>
      <td>207.0</td>
      <td>2.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>630.0</td>
      <td>205.0</td>
    </tr>
    <tr>
      <td>std</td>
      <td>10.0</td>
      <td>831.0</td>
      <td>8.0</td>
      <td>139.0</td>
      <td>1.0</td>
      <td>103.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>830.0</td>
      <td>135.0</td>
    </tr>
    <tr>
      <td>min</td>
      <td>18.0</td>
      <td>-1884.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-inf</td>
      <td>-inf</td>
      <td>-1334.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>32.0</td>
      <td>46.0</td>
      <td>8.0</td>
      <td>102.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>46.0</td>
      <td>102.0</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>39.0</td>
      <td>344.0</td>
      <td>15.0</td>
      <td>171.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>344.0</td>
      <td>171.0</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>48.0</td>
      <td>966.0</td>
      <td>21.0</td>
      <td>280.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>966.0</td>
      <td>280.0</td>
    </tr>
    <tr>
      <td>max</td>
      <td>70.0</td>
      <td>3415.0</td>
      <td>31.0</td>
      <td>639.0</td>
      <td>6.0</td>
      <td>871.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>3415.0</td>
      <td>547.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>



## <a id = "exploracao_y">4. Análise Explorátoria II </a>
Nessa seção voltaremos a fazer uma nova análise exploratória dos dados:
- para cada classe(y)
- e verificar a correlação entre as colunas


```python
from matplotlib.gridspec import GridSpec
gs = GridSpec(3,3,bottom=200,top=201,left=200,right=201,wspace=.5,hspace=.5) ## linas x colunas
plt.figure(figsize=(10,10))

for i, feature_name in enumerate(colunas_numericas[:-1] + ['balance_new','duration_new']):
    #print(i,feature_name)
    ax = plt.subplot(gs[i])
    sns.boxplot(x='y', y = feature_name, data=dados_clean, hue = 'y')  
    plt.legend()
    ax.set_xlabel('')
    ax.set_title(str(feature_name))
    
```


![png](output_49_0.png)


- podemos observar que as colunas `pday` e `previou`, estão ruins independentes da classe.
- podemos observar que as colunas `duration_new` e `balance_new` possuem valores discrepantes pra classe `NO`.

<font color='red'> Matriz de Correlação </font>

Nessa seção será verificada a correlação das variáveis, utilizando o método de pearson.


```python
plt.figure(figsize=(12,10))
chart = sns.heatmap(dados_clean.drop(columns=['balance','duration','balance_log','duration_log']).corr(),
                             linewidths=0.1, square=True, annot=True, fmt='.2f', annot_kws={"size": 9})

plt.show()
```


![png](output_51_0.png)


- podemos observar que há algumas correlações mais fortes mas negativas nas colunas `marital x age`,`poutcome x pdays`
- há algumas correlações positivas mas não tão fortes
- podemos observar que há uma certa correlação entra a variável `duration x y`. O que pode indicar que essa variável é determinante para indicar se o cliente vai aderir ou não ao produto.
- no mais, não há outras variáveis que faça correlação forte com a classe (Y)


```python
qtd_y_original = dados['y'].value_counts()
qtd_y_limpo = dados_clean['y'].value_counts()

print('Y original: \n{} - \n\nY limpa: \n{}'.format(qtd_y_original, qtd_y_limpo))

fig, axs = plt.subplots(1,2, figsize = (15,5))
qtd_y_original.plot.pie(title = 'Distribuição da Classe (base original)',
                      autopct='%.1f%%', startangle=90, ax = axs[0])

qtd_y_limpo.plot.pie(title = 'Distribuição da Classe(base limpa)',
                      autopct='%1.1f%%', startangle=90, ax = axs[1])

plt.show()
```

    Y original: 
    no     35903
    yes     4787
    Name: y, dtype: int64 - 
    
    Y limpa: 
    no     28546
    yes     2594
    Name: y, dtype: int64
    


![png](output_53_1.png)


## <a id = "conclusao">4. Conclusão</a>

- até o momento fizemos os seguintes procedimentos:
    - identificação do comportamento das variáveis numéricas, que há bastante ouliter
    - identificação dos valores para as variáveis categóricas, que há um certo desbalanceamento
    - tratamento dos dados, eliminando outlier
    - conversão das variáveis qualitativas em quantitativas
    - as colunas `balance` e `previous` tem um comportamento muito ruim, então, podem ser excluídas.
    - como identificado anteriormente, a nossa base teve uma diminuição de registros mas a classe que teve maior perda foi a `yes`, que é a nossa classe alvo positiva.
        - esse fator pode interferir no algoritmo de classificação.
        - caso seja necessário, voltaremos para a etapa de tratamento dos dados e ao invés de excluir os outlier, utilizar alguma métrica pra correção do mesmo.


```python
## SALVANDO DADOS TRATADOS EM UMA NOVA BASE
dados_clean.drop(columns=['balance','previous','balance_log','duration_log']).to_csv('data/dados_limpos.csv',index_label=False)
```

## <a id = "proxima">5. Próxima Etapas</a>
- Realizar análise dos dados para extrair inforamções de negócio. <a href="https://github.com/gabriellimagomes15/data-insight-assessment" >Análise de Dados</a>
- Desenvolvimento de algoritmo de classificação. <a href="https://github.com/gabriellimagomes15/data-insight-assessment" >Modelo ML classificação</a>


```python

```
