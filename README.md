## Classificador de Raças de Gatos
*Este é um projeto que consiste em um classificador de imagens de gatos que determina a raça do gato na imagem. O projeto utiliza uma rede neural convolucional (CNN) treinada com imagens de três raças de gatos: Laranja, Siamês e Sphinx.*
## Funcionamento
*O projeto é dividido em duas partes:*

### Treinamento do Modelo:
*O treinamento do modelo é realizado em um arquivo Python denominado model.py.*

*Neste arquivo, as imagens de treinamento e teste são carregadas e pré-processadas usando a biblioteca TensorFlow.*

*Após o treinamento, o modelo é salvo como modelo_rede_neural.h5.*
### Classificação de Imagens:

*A classificação de imagens é realizada em um arquivo Python denominado classification.py.*

*O modelo treinado (modelo_rede_neural.h5) é carregado.*

*Uma interface gráfica é apresentada ao usuário, permitindo que ele selecione uma imagem de um gato.*

*A imagem selecionada é preprocessada e passada para o modelo para fazer a predição da raça do gato.*

## Requisitos
1. Python 3.x
2. TensorFlow
3. NumPy
4. Pillow (PIL)
5. Matplotlib
6. Tkinter (para a interface gráfica)
## Como Usar
### Treinamento do Modelo:

*Execute o arquivo model.py para treinar o modelo.*

*O modelo treinado será salvo como modelo_rede_neural.h5.*

### Classificação de Imagens:

*Execute o arquivo classification.py.*

*Uma interface gráfica será aberta. Selecione uma imagem de um gato para classificar.
A raça prevista será exibida na janela da interface gráfica.*

### Projeto realizado pelos alunos:
1. [Ruan Felipe de Lima](https://github.com/RuanLima23)
2. [Lucas Gabriel](https://github.com/Lucas-gps)
3. Nicolas
4. [Bruno Cassias](https://github.com/BrunoECB)

*Do curso de Engenharia de Software, 5ª fase da Universidade do Contestado(UnC).*