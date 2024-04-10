clear; close all; clc;

h = 0.01;

zsig = -5:h:5; zsig(end) = [];
zlu = -2:h:2; zlu(end) = [];

% Sigmoidal
sigmoid = 1./(1 + exp(-zsig));
tanhsigmoid = tanh(zsig);

% Linear Unit
relu = max(0, zlu);
lrelu = max(0.05*zlu, zlu);

figure
plot(zsig, sigmoid, 'r', zsig, tanhsigmoid, 'b')
legend('Sigmoid', 'Tanh')

figure
plot(zlu, relu, 'r', zlu, lrelu, 'b')
legend('ReLU', 'Leaky ReLU')

% Export data
writetable(array2table([zsig.' sigmoid.' tanhsigmoid.'], 'VariableNames', {'z', 'sigmoid', 'tanh'}), 'sigmoidal.csv', 'LineEnding', '\n')
writetable(array2table([zlu.' relu.' lrelu.'], 'VariableNames', {'z', 'relu', 'lrelu'}), 'linearunit.csv', 'LineEnding', '\n')
