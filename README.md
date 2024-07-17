# Repositório de Projetos de PLN, Machine Learning e LLM

Bem-vindo ao meu repositório de projetos de Processamento de Linguagem Natural (PLN), Machine Learning e Modelos de Linguagem de Grande Escala (LLM). Este repositório contém uma coleção de notebooks do Google Colab que exemplificam diversas técnicas e aplicações nessas áreas da Inteligência Artificial.

<div align="center">
  
| ![I'm a talking robot You can trust me](files/talking_robot.gif) |
|:--:|
| *I'm a talking robot. You can trust me.* |

</div>

## Sumário

1. [Introdução ao Processamento de Linguagem Natural](#introdução-ao-processamento-de-linguagem-natural)
2. [Classificação de Texto com Machine Learning](#classificação-de-texto-com-machine-learning)
3. [Redes Neurais Artificiais para PLN](#redes-neurais-artificiais-para-pln)
4. [Análise de Sentimentos](#análise-de-sentimentos)
5. [Fine-tunning em Modelos de Linguagem de Grande Escala](#fine-tunning-em-modelos-de-linguagem-de-grande-escala)
6. [Agentes Autônomos em LLMs](#agentes-autônomos-em-llms)
7. [Hugging Face, BERT e Transformers](#hugging-face-bert-e-transformers)



## Introdução ao Processamento de Linguagem Natural

Este projeto aborda os conceitos básicos de Processamento de Linguagem Natural (PLN), incluindo tokenização, remoção de stopwords, stemming e lematização. O notebook demonstra como essas técnicas podem ser aplicadas em textos simples para preparar dados textuais para tarefas de Machine Learning.

[NLP com a biblioteca NLTK](NLP_com_NLTK.ipynb)

[NLP com a biblioteca spaCy](NLP_com_spaCy.ipynb)



## Classificação de Texto com Machine Learning

Neste projeto, são explorados algoritmos de Machine Learning para a classificação de textos. Utilizando um conjunto de dados de textos de emails para treinar um modelo de classificação e verificar seu desempenho com as métricas de avaliação.

[Classificação de emails em spam e não spam](Spam_email_classification_ML.ipynb)



## Redes Neurais Artificiais para PLN

Implementação de uma Rede Neural Artificial para Processamento de Linguagem Natural utilizando scikit-learn e Keras. E também uma versão de Rede Neural com vetores de alta dimensão (embeddings).

[Criação de rede neural artificial](Implementação_de_rede_neural.ipynb)

[Criação de rede neural artificial com embeddings](Implementação_de_rede_neural_com_embeddings.ipynb)



## Análise de Sentimentos

Nestes notebooks, foram utilizadas técnicas diferentes para realizar a análise de sentimentos em uma base de dados com textos de tweets. LSTM (um tipo de rede neural recorrente) e a biblioteca VADER foram utilizadas.

[Análise de sentimentos com LSTM](notebooks/analise_sentimentos_bert.ipynb)

[Análise de sentimentos a base de Regras (VADER)](notebooks/analise_sentimentos_bert.ipynb)

[Análise de sentimentos: supervisionado x regras](notebooks/analise_sentimentos_bert.ipynb)



## Fine-tunning em Modelos de Linguagem de Grande Escala

Exploração de fine-tunning em LLM realizando ajustes em um modelo pré-treinado para uma nova tarefa específica, utilizando um conjunto de dados menor e específico para a tarefa.

[Implementando fine-Tuning em um LLM usando BERT](notebooks/geracao_texto_gpt3.ipynb)



## Agentes Autônomos em LLMs

Aplicando agente autônomo para usar a capacidade de processamento de linguagem natural dos LLMs para realizar uma variedade de tarefas de maneira mais eficiente e inteligente.

[Construção de agente autônomo para LLM](notebooks/geracao_texto_gpt3.ipynb)


## Hugging Face, BERT e Transformers

Projetos que exploram os módulos disponíveis na biblioteca da empresa 🤗 Hugging Face. Aplicações poderosas no domínio da Inteligência Artificial, porém com alto nível de abstração.

[Perguntas e respostas com modelo de LLM da Hugging Face](Perguntas_e_respostas_com_Transformers.ipynb)

[Preenchimento de lacunas com BERTimbau](Preenchimento_de_lacunas_com_BERTimbau.ipynb)

---

### 🔗 Referências e Contribuições

Boa parte dos códigos são práticas realizadas em tutoriais e cursos. Segue abaixo lista de links que foram consultados e links de indicação de conteúdo que podem ser utilizados para um estudo mais abrangente.

+ [Formação Processamento de Linguagem Natural e LLM (Udemy)]([notebooks/geracao_texto_gpt3.ipynb](https://www.udemy.com/course/formacao-processamento-de-linguagem-natural-nlp/?couponCode=THANKSLEARNER24))
+ [PROF. FABIO SANTOS (YouTube)](https://www.youtube.com/@Prof.FabioSantos)

### Licença

Este projeto não está licenciado.

