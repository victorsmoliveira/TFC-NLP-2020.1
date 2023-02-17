# TFC-NLP-2020.1

Repositório com o código referente ao trabalho de conclusão de curso, sobre NLP.

Autores: Victor De Simone Oliveira, Paloma Castrioto Ribeiro
Ano: 2020

## Visão Geral

### Arquivos

Neste projeto, há uma pasta com os códigos, `code` e uma outra com os datasets, `data`.

### Código

O código principal é o [main.py](code/main.py).

Há um outro arquivo nessa pasta, [main_old.py](code/main_old.py), que não está atualizado (não está funcionando adequadamente) mas contém outras tentativas de técnicas usadas ao longo do projeto, portanto há grande valor.

### Instruções para rodar o projeto

Instalar a ferramenta [poetry](https://python-poetry.org/docs/#installation).

Com a linha de comando, entre na pasta do projeto e rode:

```powershell
> poetry install
```

Isso vai criar um virtual env e fazer o download de todas as dependências do projeto para ele.

Depois disso, é necessário fazer o download de um "dicionário" da biblioteca Spacy. Rode:

```powershell
> python -m spacy download en
```

Tenha certeza de que seu virtual env criado pelo Poetry está ativado e rode o código principal:

```powershell
> py code/main.py
```
