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
6. [Modelos GPT da OpenAI](#modelos-gpt-da-openai)
7. [Modelo BERT e Variações](#modelo-bert-e-variações)
8. [Hugging Face e Transformers](#hugging-face-e-transformers)



## Introdução ao Processamento de Linguagem Natural

Este projeto aborda os conceitos básicos de Processamento de Linguagem Natural (PLN), incluindo tokenização, remoção de stopwords, stemming e lematização. O notebook demonstra como essas técnicas podem ser aplicadas em textos simples para preparar dados textuais para tarefas de Machine Learning.

[NLP com a biblioteca NLTK](NLP_com_NLTK.ipynb)

[NLP com a biblioteca spaCy](NLP_com_spaCy.ipynb)

[Pré-processamento de dados de texto](Pre_processamento_com_NLTK_e_spaCy.ipynb)



## Classificação de Texto com Machine Learning

Neste projeto, são explorados algoritmos de Machine Learning para a classificação de textos. Utilizando um conjunto de dados de textos de emails para treinar um modelo de classificação e verificar seu desempenho com as métricas de avaliação.

[Classificação de emails em spam e não spam](Spam_email_classification_ML.ipynb)



## Redes Neurais Artificiais para PLN

Implementação de uma Rede Neural Artificial para Processamento de Linguagem Natural utilizando scikit-learn e Keras. E também uma versão de Rede Neural com vetores de alta dimensão (embeddings).

[Criação de rede neural artificial](Implementação_de_rede_neural.ipynb)

[Criação de rede neural artificial com embeddings](Implementação_de_rede_neural_com_embeddings.ipynb)

[Criando uma LSTM para PLN na prática](LSTM_simples_na_prática.ipynb)



## Análise de Sentimentos

Nestes notebooks, foram utilizadas técnicas diferentes para realizar a análise de sentimentos em uma base de dados com textos de tweets. LSTM (um tipo de rede neural recorrente) e a biblioteca VADER foram utilizadas.

[Análise de sentimentos com LSTM](Analise_de_sentimentos_com_LSTM.ipynb)

[Análise de sentimentos com VADER](Analise_de_sentimentos_com_VADER.ipynb)

[Análise de sentimentos: supervisionado x regras](Analise_de_sentimentos_supervisionado_x_regras.ipynb)



## Fine-tunning em Modelos de Linguagem de Grande Escala

Exploração de fine-tunning em LLM realizando ajustes em um modelo pré-treinado para uma nova tarefa específica, utilizando um conjunto de dados menor e específico para a tarefa.

[Implementando fine-Tuning em um LLM usando BERT](implementando_fine_tuning_em_LLM_usando_BERT.ipynb)

[Fine-tunning na prática com GPT](Fine_tunning_na_pratica_com_GPT.ipynb)

[Implementando LoRA](Implementando_LoRA.ipynb)


## Modelos GPT da OpenAI

Aplicações utilizando os modelos disponíveis na OpenAI a partir de uma chave para consulta da API. Com os modelos é possível realizar várias operações relacionadas a Modelos de Linguagem de Grande Escala.

[Testando o modelo GPT da OpenAI](Testando_modelo_GPT_da_OpenAI.ipynb)

[GPT na prática com GPTNeo](GPT_exemplo_com_GPTNeo.ipynb)

[Construção de agente autônomo para LLM](Construção_de_agente_autônomo_para_LLM.ipynb)




## Modelo BERT e Variações

Notebooks utilizando os modelos BERT disponíveis na biblioteca da 🤗 Hugging Face. O modelo é poderoso, porém as aplicações nos exemplos possuem um grau de entendimento fácil.

[Modelagem de tópicos com BERT](Modelagem_de_tópicos_com_BERT.ipynb)

[Preenchimento de lacunas com BERTimbau](Preenchimento_de_lacunas_com_BERTimbau.ipynb)

[Preenchimento de lacunas com RoBERTa](Preenchimento_de_lacunas_com_RoBERTa.ipynb)



## Hugging Face e Transformers

Projetos que exploram os módulos disponíveis na biblioteca da empresa 🤗 Hugging Face. Aplicações poderosas no domínio da Inteligência Artificial, porém com alto nível de abstração.

[Perguntas e respostas com modelo de LLM da Hugging Face](Perguntas_e_respostas_com_Transformers.ipynb)

[Resumo de textos com modelo de LLM da Hugging Face](Resumo_de_textos_com_Transformers.ipynb)

[Geração de textos com modelo de LLM da Hugging Face](Geração_de_texto_com_Transformers.ipynb)

[Transformers com T5 na prática para resumo de textos](Transformers_com_T5_na_pratica.ipynb)





---

### 🔗 Referências e Contribuições

Boa parte dos códigos são práticas realizadas em tutoriais e cursos. Segue abaixo lista de links que foram consultados e links de indicação de conteúdo que podem ser utilizados para um estudo mais abrangente.

+ [Formação Processamento de Linguagem Natural e LLM (Udemy)](https://www.udemy.com/course/formacao-processamento-de-linguagem-natural-nlp/?couponCode=THANKSLEARNER24)
+ [LLMs: Dommine GPT, Gemini, BERT e Muito Mais - 2024 (Udemy)](https://www.udemy.com/course/domine-llm/?couponCode=KEEPLEARNING)
+ [PROF. FABIO SANTOS (YouTube)](https://www.youtube.com/@Prof.FabioSantos)

### Licença

Este projeto não está licenciado.

