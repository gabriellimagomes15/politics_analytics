# Discursos na Camâra dos Deputados Federais

## Breve descrição da metodologia

Script para acessar todos os discursos dos deputados em plenário que estão disponíveis no setor de Notas Taquigráficas do site da Câmara dos Deputados.

O script permite consultar período e deputados específicos, sessão em plenário ou comissão.

## Conteúdo do repositório

### Diretório `code`

Contém os arquivos `.ipynb` que foram usados para raspar os dados do site da Câmara.

Devem ser executados na seguinte ordem:

1. `pega-links.ipynb`: executa uma busca textual para encontrar os discursos e salva os URLs onde eles estão armazenados

2. `pega-pronunciamentos-html.ipynb`: Usa o csv gerados em `pega-links` para extrair, via BeautifulSoup, os pronunciamentos para os quais existem conteúdo em HTML. Também há a opção de salvar PDFs para pronunciamentos que não foram transcritos, ainda.

3. `parseia-json.ipynb`: Usa o arquivo com os discursos **já checado manualmente**. Transforma esses dados em um arquivo .json para gerar as visualizações de dados.

### Diretório `data`

`tables`: Ao executar `pega-links`, nesse diretório serão salvos os arquivos .csv com os links que precisam ser raspados.

`txts`: Ao executar `pega-pronunciamentos-html`, esse diretório será preenchido com os discursos raspados do site da Câmara.

`csvs`: Diversos csvs gerados ao longo do fluxo de trabalho são salvos aqui, a saber: 

  - Os discursos formatados em `formata-texto`
  - Os discrusos classificados por `buscador-da-ditadura`


### Este script foi baseado no repositório do Estadão: https://github.com/estadao/bolsonaro-e-ditadura-no-congresso 