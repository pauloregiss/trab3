% Algoritmo em OCTAVE// Paulo Régis P. Lima
clear, clc
for rodada = 1:2

    % Obter o conjunto de amostras de treinamento
    dados = load('dados_de_treinamento.dat');
    quantidade_dados_de_treinamento = size(dados, 1);
    dados = [linspace(-1, -1, quantidade_dados_de_treinamento)' dados];
    x = dados(:, 1:5);

    % Associar a saída desejada para cada amostra obtida
    d = dados(:, 6:8);

    % Inicializar as matrizes de pesos W1 (Neurônios Intermediários x Entradas) e
    % W2 (Neurônios Saída x Neurônios Intermediários+1) aleatoriamente com
    % valores pequenos aleatórios
    neuronios_intermediarios = 15;
    neuronios_saida = 3;
    entradas = size(x, 2);

    if rodada == 1 % Na primeira rodada, executa o algoritmo sem momentum
        alfa = 0; % Termo momentum
        w1 = rand(neuronios_intermediarios, entradas) * 0.2;
        w2 = rand(neuronios_saida, neuronios_intermediarios + 1) * 0.2;
        titulo = 'Sem momentum: ';
        % Salvar as matrizes iniciais w1 e w2 para utilizar no algoritmo com momentum
        save('matrizes_iniciais', 'w1', 'w2');
    else % Na segunda rodada, executa o algoritmo com momentum
        alfa = 0.9; % Termo momentum
        % Carregar a matriz inicial w1 utilizada no algoritmo sem momentum
        load('matrizes_iniciais');
        titulo = 'Com momentum: ';
    end

    w1_inicial = w1;
    w1_atual = w1;
    w1_anterior = w1;
    w2_inicial = w2;
    w2_atual = w2;
    w2_anterior = w2;

    % Taxa de aprendizagem (ta) e precisão
    ta = 0.1;
    precisao = 1e-6;

    % Iniciar o contador de épocas
    ep = 1;

    % Iniciar o Erro Quadrático Médio atual
    EQM = 0;

    tic % Início da contagem do tempo de treinamento

    % Laço principal
    while true
        for k = 1:length(x)
            % Fase Forward
            I1 = w1 * x(k, :)';
            Y1 = 1 ./ (1 + exp(-I1));
            Y1 = [-1; Y1];
            I2 = w2 * Y1;
            Y2 = 1 ./ (1 + exp(-I2));

            % Fase Backward
            a = exp(-I1) ./ (1 + exp(-I1)).^2;
            b = exp(-I2) ./ (1 + exp(-I2)).^2;
            delta2 = b .* (d(k, :)' - Y2);
            w2 = w2 + (ta * delta2) * Y1' + alfa * (w2_atual - w2_anterior);
            w2_anterior = w2_atual;
            w2_atual = w2;
            delta1 = a .* (delta2' * w2(:, 2:neuronios_intermediarios + 1))';
            w1 = w1 + ta * delta1 * x(k, :) + alfa * (w1_atual - w1_anterior);
            w1_anterior = w1_atual;
            w1_atual = w1;
        end

        % Obter saída da rede ajustada
        for k = 1:length(x)
            I1 = w1 * x(k, :)';
            Y1 = 1 ./ (1 + exp(-I1));
            Y1 = [-1; Y1];
            I2 = w2 * Y1;
            Y2 = 1 ./ (1 + exp(-I2));
            EQ(:, k) = 0.5 * ((d(k, :)' - Y2).^2);
        end

        % Cálculo do EQM
        EQM = EQM + sum(EQ(:)) / length(x);
        ep = ep + 1;
        eqm(ep) = EQM;
        EQM = 0;

        if (abs(eqm(ep) - eqm(ep-1)) < precisao)
            break
        end
    end

    disp('Treinamento finalizado! ');

    tempo = toc; % Finalização da contagem do tempo de treinamento
    fprintf('Duração do treinamento: %.2f segundos \n \n', tempo);

    %% Validação
    % Obter o conjunto de dados de validação
    validacao = load('dados_de_validacao.dat');
    quantidade_dados_de_validacao = size(validacao, 1);
    validacao = [linspace(-1, -1, quantidade_dados_de_validacao)' validacao];

    xv = validacao(:, 1:5);
    yv = validacao(:, 6:8);

    % Fase Forward
    I1 = w1 * xv';
    Y1 = 1 ./ (1 + exp(-I1));
    Y1 = [-ones(1, size(xv, 1)); Y1];
    I2 = w2 * Y1;
    Y2 = 1 ./ (1 + exp(-I2));

    % Disponibilizar as saídas da rede (pós-processamento)
    Y2p = Y2 >= 0.5;

    disp('Saída sem processamento: y1 y2 y3'), Y2'
    disp('Saída processada: y1 y2 y3'), Y2p'

    % Gráfico do EQM
    figure;
    plot(eqm);
    grid on;
    title(strcat(titulo, string(tempo), ' segundos'));
    xlabel('Épocas');
    ylabel('EQM');

    if rodada == 1
        clear, clc;
    end
end

