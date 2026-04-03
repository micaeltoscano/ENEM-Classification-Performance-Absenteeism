# Classificação de Desempenho e Absenteísmo no ENEM 📊

## 📖 Sobre o Projeto
Este projeto analisa a influência de fatores socioeconômicos no desempenho e no absenteísmo dos participantes do Exame Nacional do Ensino Médio (ENEM) entre os anos de 2019 e 2023. Caracteriza-se como um estudo analítico quantitativo que utiliza técnicas de Aprendizado de Máquina (Machine Learning) para identificar padrões de desigualdade social refletidos na educação. A pesquisa utiliza microdados abertos fornecidos pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).

## 👥 Autores
* **Micael Oliveira de Lima Toscano**
* **Sérgio Cauã dos Santos**

**Orientadores:** Prof. Dr. Bruno Jefferson de Sousa Pessoa e Prof. Dr. Gilberto Farias de Sousa Filho.
**Instituição:** Universidade Federal da Paraíba (UFPB) - Centro de Informática.

## 🎯 Objetivos
* Aplicar modelos de Machine Learning para classificar o desempenho dos estudantes com base em seus dados socioeconômicos.
* Estimar a probabilidade de comparecimento dos participantes nos dias de aplicação do exame.
* Construir uma pipeline de inferência baseada nessas probabilidades para a geração de perfis de risco educacional dos alunos.
* Utilizar as previsões de desempenho como um escore para estimar chances relativas de ingresso no ensino superior.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Modelos de Machine Learning:** Decision Tree, Random Forest, Support Vector Machine (SVM) e Redes Neurais.
* **Manipulação de Dados:** Pandas, NumPy.
* **Modelagem e Avaliação:** Scikit-learn, Keras.
* **Visualização:** Matplotlib.

## ⚙️ Metodologia
1. **Engenharia de Dados:** Devido ao grande volume de informações, os dados originais do INEP foram convertidos do formato `.csv` para `.parquet`, garantindo maior eficiência e velocidade nas operações de leitura e compressão.
2. **Pré-processamento:** Foram aplicadas técnicas como *One Hot Encoding* para variáveis categóricas e *Ordinal Encoder* para preservar relações de ordem. Colunas sensíveis, como raça e sexo, foram removidas para evitar vieses sociais ou discriminação nos resultados. A variável alvo `FALTOU` foi criada para mapear a presença.
3. **Treinamento e Validação:** Os modelos passaram por validação cruzada para garantir generalização. Diferentes abordagens de divisão temporal foram testadas, incluindo o uso dos anos de 2019 a 2022 para treino/validação e o ano de 2023 exclusivamente para teste.

## 📈 Principais Resultados
* **Classificação de Desempenho:** O modelo de **Redes Neurais** demonstrou maior robustez, alcançando uma acurácia de **71%** na previsão do desempenho dos estudantes.
* **Classificação de Presença (Absenteísmo):** Modelos baseados em árvores (**Decision Tree e Random Forest**) apresentaram resultados superiores para a previsão de presença. A escolha por esses modelos também é justificada pela sua alta interpretabilidade, permitindo entender claramente as variáveis que colocam um aluno "em risco" de ausência.
* **Conclusão Geral:** O questionário socioeconômico demonstrou ser uma base valiosa para predizer o comportamento dos estudantes com aproximadamente 70% de acurácia, ajudando a identificar perfis de vulnerabilidade antes mesmo da aplicação da prova.
