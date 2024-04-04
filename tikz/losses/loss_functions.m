clear; close all; clc;

h = 0.01;

zreal = -1:h:1; zreal(end) = [];
zlog = -1:h:1; zlog(end) = [];
zhinge0 = -1.25:h:1; zhinge0(end) = [];
zhinge1 = -1:h:1.25; zhinge1(end) = [];
zprob = 0:h:1; zprob(end) = [];

% Regression: y in Reals
y = 0;
mse = 2 * (zreal - y).^2;
mae = 2 * abs(zreal - y);

% Classification y in Reals
y = -1;
logistic0 = log(1 + exp(-y * zlog));
hinge0 = max(0, 1 - y * zhinge0)/2; 

y = 1;
logistic1 = log(1 + exp(-y * zlog));
hinge1 = max(0, 1 - y * zhinge1)/2; 

% Classification y in [0, 1]
y = 0;
crossentropy0 = -(y * log(zprob) + (1 - y) * log(1 - zprob))/10;

y = 1;
crossentropy1 = -(y * log(zprob) + (1 - y) * log(1 - zprob))/10;

figure
plot(zreal, mse)
hold on
plot(zreal, mae)
legend("MSE", "MAE")
xlim([-1.25 1.25])
ylim([0 2])

figure
plot(zlog, logistic0)
hold on
plot(zlog, logistic1)
legend("Logistic 0", "Logistic 1")
xlim([-1.25 1.25])
ylim([0 2])

figure
plot(zhinge0, hinge0)
hold on
plot(zhinge1, hinge1)
legend("Hinge -1", "Hinge 1")
xlim([-1.25 1.25])
ylim([0 2])

figure
plot(zprob, crossentropy0)
hold on
plot(zprob, crossentropy1)
legend("Cross-Entropy 0", "Cross-Entropy 1")
xlim([-0.25 1.25])
ylim([0 1])

% Export data
writetable(array2table([zreal.' mse.' mae.'], 'VariableNames', {'z', 'mse', 'mae'}), 'regression.csv', 'LineEnding', '\n') 
writetable(array2table([zlog.' logistic0.' logistic1.'], 'VariableNames', {'z', 'logistic0', 'logistic1'}), 'logistic.csv', 'LineEnding', '\n') 
writetable(array2table([zhinge0.' zhinge1.' hinge0.' hinge1.'], 'VariableNames', {'z0', 'z1', 'hinge0', 'hinge1'}), 'hinge.csv', 'LineEnding', '\n') 
writetable(array2table([(zprob-0.5).' crossentropy0.' crossentropy1.'], 'VariableNames', {'z', 'crossentropy0', 'crossentropy1'}), 'crossentropy.csv', 'LineEnding', '\n') 