# Sistema de Reconhecimento Facial com Python

Este projeto é um exemplo de um sistema de reconhecimento facial desenvolvido do zero utilizando Python e o Google Colab. O sistema integra duas funcionalidades principais:

- **Detecção de Faces:** Utiliza a biblioteca [MTCNN](https://github.com/ipazc/mtcnn) para identificar e localizar faces em imagens capturadas pela webcam.
- **Reconhecimento/ Classificação de Faces:** Aplica uma rede neural simples (CNN) construída com TensorFlow/Keras para classificar as faces detectadas.  
  > **Atenção:** O modelo de classificação apresentado neste projeto é ilustrativo e não foi treinado. Para obter um sistema funcional, é necessário treinar o modelo com um conjunto de dados real e rotulado.

## Funcionalidades

- **Captura de Imagem via Webcam:**  
  Implementa uma função com JavaScript integrada ao Colab para capturar uma foto utilizando a webcam.

- **Detecção de Faces:**  
  Utiliza o MTCNN para identificar e delimitar faces na imagem capturada.

- **Classificação de Faces:**  
  Define uma CNN simples que classifica cada face detectada em duas categorias: "Pessoa" e "Desconhecido".  
  *Nota:* O modelo serve apenas para demonstrar o fluxo de processamento e reconhecimento.

- **Visualização dos Resultados:**  
  Exibe a imagem final com as faces delimitadas por retângulos e os respectivos rótulos de classificação.

## Dependências

As principais bibliotecas utilizadas são:

- [TensorFlow](https://www.tensorflow.org/)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [OpenCV](https://opencv.org/) (versão headless para Colab)
- [NumPy](https://numpy.org/)
- [Google Colab](https://colab.research.google.com/)

Para instalar as dependências, execute:

```bash
!pip install mtcnn opencv-python-headless tensorflow
