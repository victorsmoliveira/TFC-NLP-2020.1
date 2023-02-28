# TFC-NLP-2020.1

Repositório com o código referente ao trabalho de conclusão de curso, sobre NLP.

Autores: Victor De Simone Oliveira, Paloma Castrioto Ribeiro
Ano: 2020

## Visão Geral

### Arquivos

Neste projeto, há algumas pastas:

- `code`: contém o código principal do projeto
- `data`: contém os datasets
- `example_results`: contém um exemplo de output.

Além disso, há alguns arquivos de configuração na raiz do projeto, e um pdf contendo o TFC.

### Código

O código principal é o [main.py](code/main.py).

Há um outro arquivo nessa pasta, [main_old.py](code/main_old.py), que não está atualizado (não está funcionando adequadamente) mas contém outras tentativas de técnicas usadas ao longo do projeto, portanto há grande valor. Além disso, é nele que está contido o código que gera as imagens/visualizações encontradas no TCC. Esse código deve ser incorporado ao principal, ou colocado em um novo dedicado para isso (TODO).

### Instruções para rodar o projeto

Instale a ferramenta [poetry](https://python-poetry.org/docs/#installation).

Com a linha de comando, entre na pasta do projeto e rode:

```powershell
> poetry install
```

Isso vai criar um virtual env e fazer o download de todas as dependências do projeto para ele.

Ative o virtual env criado pelo Poetry:

```powershell
> poetry shell
```

Depois disso, é necessário fazer o download de um "dicionário" da biblioteca Spacy. Rode:

```powershell
> python -m spacy download en
```

E depois rode o código principal:

```powershell
> py code/main.py
```

### Output

O output do código são dois arquivos, um CSV com o nome de `results.csv` e um Excel com o nome de `results.xlsx`, que vão ser salvos na raiz do projeto.

Um exemplo de output pode ser encontrado na pasta `example_results`.
