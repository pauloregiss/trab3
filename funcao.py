from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot as plt
import time

# Carregar o conjunto de amostras de treinamento
dados = np.loadtxt('dados_de_treinamento.dat')
quantidade_dados_de_treinamento = dados.shape[0]
dados = np.hstack((-np.ones((quantidade_dados_de_treinamento, 1)), dados))
x = dados[:, :5]
d = dados[:, 6:9]

# Inicializar os pesos W1 (neuronios_intermediarios x entradas) e W2 (neuronios_saida x neuronios_intermediarios+1) aleatoriamente com valores pequenos
neuronios_intermediarios = 15
neuronios_saida = 3
entradas = x.shape[1]

rodada = 1

if rodada == 1:  # Na primeira rodada, executa o algoritmo sem momentum
    alfa = 0  # Termo momentum
    np.random.seed(0)  # Define uma semente para reproduzibilidade
    w1 = np.random.uniform(0, 0.2, (neuronios_intermediarios, entradas))
    w2 = np.random.uniform(0, 0.2, (neuronios_saida, neuronios_intermediarios + 1))
    titulo = 'Sem momentum: '
    # Salvar as matrizes iniciais w1 e w2 para utilizar no algoritmo com momentum
    np.savez('matrizes_iniciais.npz', w1=w1, w2=w2)
else:  # Na segunda rodada, executa o algoritmo com momentum
    alfa = 0.9  # Termo momentum
    # Carregar a matriz inicial w1 utilizada no algoritmo sem momentum
    matrizes_iniciais = np.load('matrizes_iniciais.npz')
    w1 = matrizes_iniciais['w1']
    w2 = matrizes_iniciais['w2']
    titulo = 'Com momentum: '

w1_inicial = w1.copy()
w1_atual = w1.copy()
w1_anterior = w1.copy()
w2_inicial = w2.copy()
w2_atual = w2.copy()
w2_anterior = w2.copy()

# Taxa de aprendizagem (ta) e precisão
ta = 0.1
precisao = 1e-6

# Iniciar o contador de épocas
ep = 1

# Iniciar o Erro Quadrático Médio atual
EQM = 0

# Laço principal
while True:
    EQ = []
    for k in range(len(x)):
        # Fase Forward
        I1 = np.dot(w1, x[k, :])
        Y1 = 1 / (1 + np.exp(-I1))
        Y1 = np.hstack((-1, Y1))
        I2 = np.dot(w2, Y1)
        Y2 = 1 / (1 + np.exp(-I2))

        # Fase Backward
        a = np.exp(-I1) / (1 + np.exp(-I1)) ** 2
        b = np.exp(-I2) / (1 + np.exp(-I2)) ** 2
        delta2 = b * (d[k, :] - Y2.reshape(-1))  # Correção na dimensão de Y2
        w2 = w2 + ta * np.outer(delta2, Y1) + alfa * (w2_atual - w2_anterior)
        w2_anterior = w2_atual.copy()
        w2_atual = w2.copy()
        delta1 = a * np.dot(w2[:, 1:].T, delta2.reshape(-1, 1))  # Correção na dimensão de delta2
        w1 = w1 + ta * np.outer(delta1, x[k, :]) + alfa * (w1_atual - w1_anterior)
        w1_anterior = w1_atual.copy()
        w1_atual = w1.copy()

        # Cálculo do EQM
        EQ.append(0.5 * np.sum((d[k, :] - Y2.reshape(-1)) ** 2))
    # Calcular o EQM médio
   EQ.append(0.5 * np.sum((d[k, :] - Y2) ** 2))
    if ep > 1 and abs(EQM[ep - 1] - EQM[ep - 2]) < precisao:
        break

    ep += 1

print('Treinamento finalizado!')
print('Duração do treinamento: {:.2f} segundos\n'.format(TemporaryFile))

# Validação
validacao = np.loadtxt('dados_de_validacao.dat')
quantidade_dados_de_validacao = validacao.shape[0]
validacao = np.hstack((-np.ones((quantidade_dados_de_validacao, 1)), validacao))

xv = validacao[:, :5]
yv = validacao[:, 6:9]

# Fase Forward na rede treinada usando os dados de validação
I1 = np.dot(w1, xv.T)
Y1 = 1 / (1 + np.exp(-I1))
Y1 = np.vstack((-np.ones(xv.shape[0]), Y1))
I2 = np.dot(w2, Y1)
Y2 = 1 / (1 + np.exp(-I2))

# Processar a saída da rede para obter valores binários (0 ou 1) com base em um limiar de 0,5
Y2p = np.where(Y2 >= 0.5, 1, 0)

print('Saída sem processamento: y1 y2 y3')
print(Y2.T)
print('Saída processada: y1 y2 y3')
print(Y2p.T)

# Gráfico do EQM
plt.plot(EQM)
plt.grid()
plt.title(titulo)
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.show()
