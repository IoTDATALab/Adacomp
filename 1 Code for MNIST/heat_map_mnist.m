clear
clc
%关于不同学习率下六种算法的对比数据
test_acc = [
    15.31, 20.68, 24.51, 28.54, 33.6, 38.31, 42.37, 45.14, 47.33, 49.42; %SGD lr=0.00001 
    33.98, 49.62, 59.19, 66.06, 70.28, 73.79, 76.63, 78.9, 80.81, 82.45; %SGD lr=0.00005
    49.92, 66.32, 74.03, 79.12, 82.44, 84.81, 86.26, 87.31, 87.8, 88.55; %SGD lr=0.0001
    83.01, 88.52, 89.97, 90.89, 91.29, 91.79, 92.32, 92.55, 92.87, 93.25; %SGD lr=0.0005
    88.05, 90.86, 91.74, 92.5, 93.3, 93.71, 94.32, 94.62, 94.92, 95.28; %SGD, LR is 0.001
    92.99, 94.25, 95.97, 96.44, 96.95, 97.07, 97.52, 97.6, 97.47, 97.86; %SGD, LR is 0.005
    94.69, 94.89, 96.5, 97.45, 97.9, 98.04, 98.27, 98.14, 98.02, 98.38; %SGD, LR is 0.01
    96.12, 97.05, 97.3, 98.4, 98.41, 98.7, 98.71, 98.29, 98.7, 98.73; %SGD, LR is 0.02
    96.96, 98.09, 97.22, 98.7, 98.74, 98.68, 98.79, 98.77, 98.89, 98.93; %SGD, LR is 0.03
    97.42, 98.19, 97.99, 98.84, 98.81, 98.65, 98.83, 98.87, 98.72, 99.04; %SGD, LR is 0.04
    97.85, 98.24, 98.56, 98.76, 98.77, 98.9, 98.86, 99.02, 98.74, 98.97; %SGD, LR is 0.05
    98.03, 98.31, 98.71, 98.92, 98.63, 98.69, 98.85, 99.08, 98.69, 99.03; %SGD, LR is 0.06
    98.21, 98.46, 98.74, 98.88, 98.77, 98.95, 99.02, 99.07, 98.93, 99.01; %SGD, LR is 0.07
    98.25, 98.58, 98.95, 98.89, 98.84, 98.75, 98.98, 99.08, 98.85, 99.05; %SGD, LR is 0.08
    98.41, 98.56, 98.75, 98.83, 98.66, 98.73, 99.02, 98.99, 98.89, 99.06; %SGD, LR is 0.09
    98.52, 98.79, 98.86, 98.94, 98.69, 98.86, 98.96, 99.04, 98.92, 99.06; %SGD, LR is 0.1
    98.64, 98.04, 98.78, 98.96, 98.81, 99.0, 99.07, 98.99, 99.04, 99.14; %SGD, LR is 0.2
    98.42, 97.64, 98.86, 98.83, 98.87, 98.91, 98.86, 99.01, 98.95, 98.82; %SGD, LR is 0.3
    98.4, 98.69, 98.72, 98.63, 98.59, 98.84, 98.69, 98.89, 98.81, 98.96; %SGD, LR is 0.4
    97.9, 98.38, 97.11, 98.63, 98.7, 98.5, 98.61, 98.59, 98.6, 97.76; %SGD, LR is 0.5
    97.03, 97.53, 98.4, 98.18, 98.36, 97.68, 98.29, 98.45, 98.12, 98.51; %SGD, LR is 0.6
    97.49, 96.86, 97.56, 98.23, 98.19, 97.33, 98.33, 97.98, 97.77, 98.08; %SGD, LR is 0.7
    93.9, 96.47, 96.52, 97.75, 97.94, 98.01, 97.79, 98.12, 98.06, 97.35; %SGD, LR is 0.8
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35; %SGD, LR is 0.9
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %SGD, LR is 1
    49.17, 66.03, 73.88, 79.01, 82.31, 84.73, 86.24, 87.25, 87.81, 88.46; %Momenum, lr=0.00001
    82.44, 88.32, 89.96, 90.83, 91.38, 91.79, 92.24, 92.48, 92.84, 93.32; %Momenum, lr=0.00005
    88.17, 90.75, 91.79, 92.43, 93.24, 93.68, 94.24, 94.71, 94.84, 95.52; %Momenum, lr=0.0001
    92.63, 94.92, 95.71, 96.56, 96.97, 97.48, 97.57, 97.36, 97.74, 97.81; %Momenum, lr=0.0005
    94.08, 96.49, 96.95, 97.66, 97.98, 98.27, 98.24, 98.1, 98.26, 98.42; %Momenum, lr=0.001
    97.71, 98.44, 98.38, 98.74, 98.89, 99.0, 98.93, 98.75, 98.89, 99.06; %Momenum, lr=0.005
    98.37, 98.48, 98.73, 98.94, 98.99, 99.09, 98.97, 98.9, 98.96, 98.99; %Momenum, lr=0.01
    98.47, 98.62, 98.89, 98.92, 99.03, 98.88, 98.95, 98.95, 99.03, 99.0; %Momenum, lr=0.02
    98.63, 98.58, 98.93, 98.79, 98.92, 98.98, 99.12, 98.89, 99.0, 98.97; %Momenum, lr=0.03
    98.39, 98.02, 98.94, 98.39, 98.84, 98.86, 98.8, 98.7, 98.65, 98.18; %Momenum, lr=0.04
    98.33, 97.71, 98.47, 98.48, 98.77, 98.91, 98.74, 98.86, 98.88, 98.55; %Momenum, lr=0.05
    98.37, 98.57, 98.74, 98.54, 98.34, 98.12, 98.44, 98.95, 98.79, 98.5;
    98.39, 98.08, 98.5, 98.53, 98.52, 98.71, 98.91, 98.6, 98.79, 98.06;
    97.92, 98.14, 98.47, 97.95, 98.24, 98.61, 97.92, 98.52, 98.2, 98.33;
    98.11, 97.78, 98.23, 98.4, 98.16, 98.01, 98.03, 98.39, 98.44, 98.03;
    97.75, 97.53, 98.26, 97.73, 97.46, 97.9, 97.84, 97.06, 97.79, 97.61; %Momenum, lr=0.1
    92.74, 92.89, 94.36, 95.03, 94.03, 93.2, 94.65, 94.87, 94.9, 95.45;
    9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8;
    9.82, 11.35, 10.32, 11.35, 9.58, 10.1, 10.28, 10.09, 9.58, 11.35;
    9.82, 11.35, 10.32, 10.28, 10.32, 10.1, 10.28, 10.09, 9.58, 11.35; %Momentum, LR is 0.5
    9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8;
    10.09, 11.35, 10.32, 10.28, 11.35, 10.1, 10.28, 10.09, 9.82, 9.8;
    9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8;
    10.09, 9.74, 10.32, 11.35, 11.35, 10.1, 10.1, 9.74, 9.82, 9.8;
    9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8; %Momentum, LR is 1
    65.01, 71.19, 73.83, 75.22, 76.28, 77.56, 78.52, 79.48, 80.09, 80.77; %Adagrad, LR is 0.00001
    85.27, 87.94, 89.22, 89.89, 90.5, 90.8, 91.13, 91.34, 91.48, 91.73; %Adagrad, LR is 0.00005
    89.92, 91.39, 91.95, 92.38, 92.63, 93.0, 93.27, 93.36, 93.53, 93.74; %Adagrad, LR is 0.0001
    94.31, 95.53, 96.39, 96.73, 96.99, 97.28, 97.35, 97.52, 97.51, 97.78; %Adagrad, LR is 0.0005
    96.54, 97.53, 97.79, 98.16, 98.23, 98.35, 98.33, 98.44, 98.4, 98.51; %Adagrad, LR is 0.001
    98.43, 98.73, 98.78, 98.93, 98.98, 98.96, 98.98, 98.98, 98.87, 98.95; %Adagrad, LR is 0.005
    98.4, 98.71, 98.68, 98.85, 98.91, 98.93, 98.9, 98.86, 98.82, 98.81; %Adagrad, LR is 0.01
    98.33, 98.62, 98.73, 98.87, 98.94, 98.83, 98.98, 98.9, 98.83, 98.91;
    98.26, 98.68, 98.84, 98.97, 98.88, 98.95, 98.98, 98.89, 98.96, 98.92;
    98.17, 98.58, 98.59, 98.73, 98.77, 98.76, 98.86, 98.93, 98.63, 98.81
    98.08, 98.43, 98.52, 98.48, 98.63, 98.65, 98.69, 98.73, 98.74, 98.68; %Adagrad, LR is 0.05
    98.03, 98.45, 98.39, 98.58, 98.65, 98.66, 98.6, 98.66, 98.69, 98.65;
    97.96, 98.26, 98.2, 98.41, 98.49, 98.54, 98.68, 98.65, 98.61, 98.65;
    97.42, 98.16, 98.39, 98.48, 98.5, 98.69, 98.65, 98.64, 98.59, 98.72;
    95.5, 97.37, 97.99, 98.26, 98.26, 98.43, 98.36, 98.48, 98.43, 98.44;
    96.37, 97.54, 98.2, 98.25, 98.53, 98.24, 98.49, 98.52, 98.54, 98.6; %Adagrad, LR is 0.1
    11.35, 11.35, 92.94, 95.19, 96.31, 96.3, 96.33, 96.51, 96.96, 96.59;
    11.36, 11.37, 11.37, 11.47, 72.24, 83.76, 87.31, 88.88, 88.96, 90.28;
    11.35, 11.35, 11.35, 11.35, 11.35, 10.08, 11.35, 11.36, 11.36, 11.98;
    11.35, 11.35, 11.36, 11.36, 11.35, 10.07, 11.36, 11.42, 11.69, 11.86; %Adagrad, LR is 0.5
    8.92, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 10.1;
    11.35, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35;
    11.35, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35;
    8.92, 10.28, 10.1, 10.1, 11.35, 10.1, 9.8, 11.35, 10.1, 10.1;
    10.32, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35; %Adagrad, LR is 1
    91.98, 93.54, 95.02, 95.89, 96.48, 96.94, 97.25, 97.46, 97.61, 97.87; %RMSprop, LR is 0.00001
    96.15, 97.7, 98.03, 98.38, 98.65, 98.74, 98.79, 98.75, 98.68, 98.86; %RMSprop, LR is 0.00005
    97.59, 98.29, 98.44, 98.79, 98.86, 98.9, 98.99, 98.86, 98.68, 98.87; %RMSprop, LR is 0.0001
    98.66, 98.73, 98.88, 99.13, 98.91, 99.01, 99.04, 98.89, 98.72, 99.1; %RMSprop, LR is 0.0005
    98.48, 98.32, 98.89, 98.83, 98.97, 99.02, 98.93, 99.06, 98.87, 98.93; %RMSprop, LR is 0.001
    97.81, 97.53, 95.93, 98.21, 98.34, 98.52, 98.4, 98.14, 98.7, 98.4; %RMSprop, LR is 0.005
    96.62, 97.4, 97.67, 97.61, 98.29, 98.05, 98.17, 97.91, 97.66, 98.22; %RMSprop, LR is 0.01
    11.35, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35;
    10.32, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35;
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35;
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 10.1; %RMSprop, LR is 0.05
    8.92, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 10.1;
    8.92, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 10.1;
    8.92, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 9.74, 11.35, 10.1;
    8.92, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 9.74, 11.35, 10.1;
    8.92, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 9.74, 11.35, 10.1; %RMSprop, LR is 0.1
    8.92, 9.74, 9.58, 10.1, 10.32, 9.58, 10.28, 9.74, 11.35, 10.1;
    8.92, 10.09, 9.58, 10.1, 10.32, 9.58, 10.28, 9.74, 10.09, 10.1;
    8.92, 10.09, 9.74, 10.1, 10.32, 9.58, 10.28, 9.74, 10.09, 9.8;
    8.92, 10.09, 9.74, 10.1, 10.32, 9.58, 10.28, 9.74, 10.09, 9.8; %RMSprop, LR is 0.5
    8.92, 10.09, 10.09, 10.1, 10.32, 9.58, 10.28, 9.74, 10.09, 9.8;
    8.92, 10.09, 10.09, 10.1, 8.92, 9.58, 10.28, 9.74, 10.09, 9.8;
    8.92, 10.09, 10.09, 10.1, 8.92, 9.58, 10.28, 9.74, 10.09, 9.8;
    8.92, 10.09, 10.09, 10.1, 8.92, 9.58, 10.28, 9.74, 10.09, 9.8;
    8.92, 10.09, 10.09, 10.1, 8.92, 9.58, 10.28, 9.74, 10.09, 9.8; %RMSprop, LR is 1
    12.51, 15.05, 17.96, 20.26, 22.03, 23.51, 25.03, 26.52, 28.76, 30.74; %Adadelta, LR is 0.00001
    22.03, 30.76, 41.14, 48.66, 53.83, 58.25, 61.75, 64.55, 66.73, 68.65; %Adadelta, LR is 0.00005
    30.95, 49.07, 58.4, 64.45, 68.65, 71.74, 74.35, 76.24, 78.1, 79.73; %Adadelta, LR is 0.0001
    69.95, 80.24, 84.86, 87.31, 88.76, 89.65, 90.31, 90.74, 90.97, 91.36; %Adadelta, LR is 0.0005
    80.76, 87.41, 89.63, 90.79, 91.29, 91.77, 92.08, 92.36, 92.76, 92.91; %Adadelta, LR is 0.001
    91.26, 92.73, 93.81, 94.77, 95.26, 95.99, 96.48, 96.74, 96.97, 97.32; %Adadelta, LR is 0.005
    92.62, 94.4, 95.94, 96.73, 97.2, 97.63, 97.77, 97.91, 97.92, 98.2; %Adadelta, LR is 0.01
    94.44, 96.5, 97.58, 97.92, 98.12, 98.31, 98.38, 98.51, 98.44, 98.57;
    95.6, 97.45, 97.86, 98.26, 98.55, 98.58, 98.64, 98.55, 98.69, 98.75;
    96.37, 97.9, 98.07, 98.53, 98.74, 98.69, 98.77, 98.64, 98.64, 98.8;
    96.79, 98.15, 98.32, 98.61, 98.78, 98.73, 98.76, 98.69, 98.75, 98.9; %Adadelta, LR is 0.05
    97.18, 98.27, 98.43, 98.69, 98.89, 98.8, 98.85, 98.79, 98.79, 98.96;
    97.33, 98.41, 98.49, 98.72, 98.86, 98.8, 98.84, 98.84, 98.82, 98.94;
    97.61, 98.45, 98.54, 98.76, 98.89, 98.83, 98.86, 98.95, 98.85, 98.96;
    97.71, 98.54, 98.52, 98.74, 98.84, 98.87, 98.87, 98.87, 98.87, 98.93;
    97.9, 98.64, 98.54, 98.81, 98.86, 98.94, 98.94, 98.95, 98.88, 98.95; %Adadelta, LR is 0.1
    98.27, 98.73, 98.79, 99.02, 98.95, 98.97, 99.06, 99.05, 98.84, 99.03;
    98.53, 98.85, 98.91, 99.02, 98.92, 99.03, 99.06, 99.09, 98.97, 99.07;
    98.61, 98.78, 98.84, 99.06, 98.88, 98.93, 99.1, 99.0, 98.91, 99.07;
    98.6, 98.75, 98.8, 98.97, 98.85, 99.02, 98.98, 98.9, 99.07, 99.1; %Adadelta, LR is 0.5
    98.53, 98.62, 98.84, 98.98, 98.76, 98.86, 99.09, 98.96, 99.05, 99.15;
    98.61, 98.75, 98.9, 99.03, 98.88, 98.95, 99.06, 99.1, 99.11, 99.03;
    98.53, 98.53, 98.86, 99.07, 98.97, 98.97, 99.05, 99.11, 99.11, 99.16;
    98.54, 98.53, 98.77, 98.97, 98.96, 99.01, 99.09, 98.97, 99.06, 99.09;
    98.56, 98.58, 98.99, 98.99, 98.96, 99.07, 98.92, 99.03, 99.1, 99.1; %Adadelta, LR is 1
    91.68, 93.68, 95.04, 95.76, 96.37, 96.87, 97.26, 97.54, 97.63, 97.9; %Adam lr=0.00001 
    96.16, 97.68, 98.09, 98.31, 98.55, 98.63, 98.79, 98.76, 98.75, 98.8; %Adam lr=0.00005
    97.53, 98.32, 98.38, 98.64, 98.75, 98.86, 98.93, 98.84, 98.73, 98.86; %Adam lr=0.0001
    98.4, 97.78, 98.64, 98.56, 98.09, 98.34, 98.31, 98.37, 98.49, 98.32; %Adam lr=0.0005
    98.58, 98.75, 98.95, 98.67, 98.86, 98.74, 98.45, 98.82, 98.92, 98.93; %Adam, LR is 0.001
    92.99, 94.25, 95.97, 96.44, 96.95, 97.07, 97.52, 97.6, 97.47, 97.86; %Adam, LR is 0.005
    95.67, 95.93, 96.45, 97.04, 95.97, 97.1, 96.85, 97.45, 96.96, 97.4; %Adam, LR is 0.01
    94.71, 95.65, 93.41, 96.23, 95.72, 96.17, 96.61, 96.16, 96.0, 96.84; %Adam, LR is 0.02
    94.61, 91.71, 94.72, 93.55, 94.72, 93.77, 93.63, 94.84, 94.69, 94.39; %Adam, LR is 0.03
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.04
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.05
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.06
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.07
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.08
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.09
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.1
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.2
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.3
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.4
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.5
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.6
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.7
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.8
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 0.9
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 10.1, 11.35; %Adam, LR is 1
    86.81, 90.85, 91.82, 92.62, 93.27, 93.67, 94.26, 94.64, 94.87, 95.37; %Adamax lr=0.00001 
    93.09, 95.17, 96.34, 97.02, 97.42, 97.74, 97.9, 97.97, 98.0, 98.22; %Adamax lr=0.00005
    94.79, 96.82, 97.63, 98.02, 98.32, 98.46, 98.46, 98.59, 98.5, 98.53; %Adamax lr=0.0001
    97.93, 98.38, 98.71, 98.81, 98.82, 99.0, 99.01, 98.87, 98.86, 99.0; %Adamax lr=0.0005
    98.39, 98.71, 98.91, 98.83, 98.91, 99.08, 99.04, 98.96, 98.98, 98.93; %Adamax, LR is 0.001
    98.56, 98.87, 98.94, 98.96, 98.78, 99.1, 99.07, 98.98, 99.12, 99.14; %Adamax, LR is 0.005
    98.51, 98.66, 98.75, 98.85, 98.86, 98.77, 99.02, 98.91, 98.93, 98.67; %Adamax, LR is 0.01
    96.82, 97.9, 98.07, 98.09, 98.1, 98.41, 98.05, 98.4, 98.27, 98.32; %Adamax, LR is 0.02
    96.59, 97.33, 98.38, 98.33, 98.65, 98.64, 98.52, 98.49, 98.52, 98.59; %Adamax, LR is 0.03
    97.55, 97.86, 97.81, 98.36, 98.17, 98.16, 98.11, 98.41, 98.38, 98.49; %Adamax, LR is 0.04
    11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 10.28, 11.35, 11.35; %Adamax, LR is 0.05
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.06
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.07
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.08
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.09
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.1
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.2
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.3
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.4
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.5
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.6
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.7
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.8
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 0.9
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adamax, LR is 1
    96.93, 98.31, 98.51, 98.67, 98.64, 98.68, 98.7, 98.67, 98.71, 98.82; %Adacomp, LR is 0.00001
    97.53, 98.29, 98.44, 98.45, 98.44, 98.62, 98.78, 98.74, 98.66, 98.87; %Adacomp, LR is 0.00005
    97.57, 98.16, 98.26, 98.27, 98.37, 98.58, 98.74, 98.88, 98.69, 98.81; %Adacomp, LR is 0.0001
    97.26, 97.83, 97.9, 97.91, 97.97, 98.3, 98.43, 98.49, 98.55, 98.75; %Adacomp, LR is 0.0005
    97.65, 98.3, 98.34, 98.59, 98.58, 98.7, 98.77, 98.82, 98.83, 98.91; %Adacomp, LR is 0.001
    97.5, 97.87, 98.1, 98.08, 98.04, 98.32, 98.47, 98.59, 98.58, 98.7; %Adacomp, LR is 0.005
    97.62, 98.3, 98.41, 98.43, 98.41, 98.67, 98.79, 98.82, 98.81, 98.84; %Adacomp, LR is 0.01
    97.79, 98.37, 98.5, 98.58, 98.57, 98.73, 98.9, 98.8, 98.75, 98.83;
    97.69, 98.25, 98.29, 98.31, 98.28, 98.43, 98.57, 98.66, 98.74, 98.75;
    97.59, 98.36, 98.41, 98.41, 98.38, 98.56, 98.75, 98.8, 98.77, 98.84;
    97.69, 98.36, 98.45, 98.45, 98.46, 98.72, 98.85, 98.86, 98.84, 98.92; %Adacomp, LR is 0.05
    97.09, 98.06, 98.23, 98.18, 98.31, 98.59, 98.63, 98.72, 98.66, 98.81; 
    97.87, 98.29, 98.53, 98.53, 98.54, 98.68, 98.73, 98.75, 98.76, 98.89;
    97.86, 98.17, 98.27, 98.28, 98.26, 98.58, 98.63, 98.78, 98.75, 98.78;
    97.52, 98.0, 98.42, 98.46, 98.44, 98.66, 98.62, 98.69, 98.57, 98.75;
    97.86, 98.35, 98.5, 98.53, 98.51, 98.8, 98.89, 98.91, 98.82, 98.99; %Adacomp, LR is 0.1
    97.65, 97.98, 98.02, 98.08, 98.18, 98.38, 98.57, 98.63, 98.71, 98.69;
    97.18, 97.61, 97.93, 98.29, 98.28, 98.59, 98.73, 98.78, 98.78, 98.76;
    97.78, 97.85, 98.18, 98.3, 98.32, 98.45, 98.61, 98.65, 98.61, 98.65;
    97.45, 97.91, 98.27, 98.23, 98.33, 98.41, 98.52, 98.66, 98.58, 98.78; %Adacomp, LR is 0.5
    97.43, 98.14, 98.39, 98.39, 98.41, 98.59, 98.71, 98.77, 98.74, 98.88;
    97.14, 97.99, 98.23, 98.34, 98.38, 98.51, 98.55, 98.65, 98.55, 98.7;
    97.72, 98.24, 98.46, 98.44, 98.47, 98.42, 98.71, 98.72, 98.73, 98.81;
    97.24, 98.13, 98.16, 98.27, 98.32, 98.56, 98.63, 98.74, 98.66, 98.69;
    97.76, 98.34, 98.54, 98.63, 98.6, 98.7, 98.78, 98.82, 98.8, 98.79; %Adacomp, LR is 1   
    ];
test_acc_large_lr = [
    98.56, 98.58, 98.99, 98.99, 98.96, 99.07, 98.92, 99.03, 99.1, 99.1; %Adadelta, LR is 1
    98.43, 98.73, 98.7, 98.86, 98.42, 98.91, 99.01, 98.94, 98.98, 99.06; %Adadelta, LR is 2
    98.36, 98.4, 98.75, 98.69, 98.75, 98.77, 98.87, 98.89, 99.02, 98.86; %Adadelta, LR is 3
    98.42, 98.5, 98.67, 98.7, 98.58, 98.63, 98.88, 98.88, 98.42, 98.73; %Adadelta, LR is 4
    98.04, 98.59, 98.24, 98.6, 98.57, 98.6, 98.66, 98.72, 98.74, 98.73; %Adadelta, LR is 5
    97.45, 97.9, 98.49, 98.6, 98.4, 98.62, 98.43, 98.55, 98.54, 98.86;  %Adadelta, LR is 6
    97.67, 98.1, 98.28, 98.3, 97.76, 97.73, 98.29, 98.58, 98.5, 98.55; %Adadelta, LR is 7
    97.02, 97.02, 97.03, 97.56, 97.56, 97.63, 98.3, 97.76, 98.2, 97.13;
    96.67, 97.61, 97.35, 97.58, 97.94, 98.25, 98.47, 98.27, 98.38, 98.46;
    10.32, 11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35; %Adadelta, LR is 10
    10.32, 9.74, 11.35, 11.35, 11.35, 10.1, 9.8, 11.35, 11.35, 11.35; %Adadelta, LR is 15
    97.76, 98.34, 98.54, 98.63, 98.6, 98.7, 98.78, 98.82, 98.8, 98.79; %Adacomp, LR is 1
    97.54, 98.14, 98.44, 98.37, 98.45, 98.63, 98.73, 98.71, 98.64, 98.68; %Adacomp, LR is 2 
    97.34, 97.97, 98.02, 98.02, 98.05, 98.37, 98.55, 98.53, 98.54, 98.58; %Adacomp, LR is 3
    96.28, 97.21, 97.79, 97.86, 97.87, 98.22, 98.39, 98.56, 98.43, 98.52; %Adacomp, LR is 4 
    97.8, 98.04, 98.28, 98.27, 98.34, 98.51, 98.7, 98.75, 98.67, 98.7; %Adacomp, LR is 5 
    96.81, 97.68, 97.8, 97.85, 97.87, 98.22, 98.4, 98.57, 98.55, 98.71; %Adacomp, LR is 6
    96.45, 97.61, 97.58, 97.67, 97.84, 98.25, 98.43, 98.5, 98.28, 98.46; %Adacomp, LR is 7
    97.84, 98.02, 98.26, 98.3, 98.33, 98.5, 98.61, 98.74, 98.66, 98.82;
    97.8, 97.95, 97.94, 97.96, 98.01, 98.24, 98.47, 98.61, 98.55, 98.75;
    96.62, 97.72, 98.01, 98.14, 98.16, 98.29, 98.48, 98.58, 98.52, 98.7; %Adacomp, LR is 10
    97.46, 98.01, 98.15, 98.19, 98.25, 98.34, 98.47, 98.48, 98.55, 98.57; %Adacomp, LR is 15
    ];
figure(1)
rho = reshape(test_acc(:,end), 25, 8);
rho(rho<=25) = nan;
% 绘制热图
string_name={'1e-5','5e-5','1e-4','5e-4','1e-3','5e-3','0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'};
xvalues = string_name;
yvalues = {'SGD','Momen.','Adagrad','RMSprop','Adadelta','Adam','Adamax','Ours'};
h1 = heatmap(yvalues,xvalues, rho, 'FontSize',12, 'FontName','Times New Roman',...
    'MissingDataColor',[0.941176470588235 0.941176470588235 0.941176470588235]);
% h.ColorScaling = 'scaledcolumns';
h1.Title = 'Test accuracy';
% h1.ColorbarVisible = 'off';
% map = [1 1 1; 1 1 0; 0.5 1 0.4; 0.2 0.85 0.2; 0.4 0.7 1; 0.2 0.5 0.8]; % 自己定义颜色
colormap(hot)
% saveas(gcf,sprintf('comparison_lr.eps'),'bmp'); %保存图片

%关于Adadelta和Adacomp在较大学习率下的对比示意（图2-1）和有效区间示意（图2-2）
figure(2)
a = reshape(test_acc_large_lr(:,end),11,2);
% data_lar_lr = [99.1000000000000,98.2200000000000;99.0600000000000,97.8200000000000;
%     98.8600000000000,96.9700000000000;98.7300000000000,96.8300000000000;98.7300000000000,97; 11.35,96.95;11.35,96.58];
subplot(2,1,1)
h1 = plot([1,2,3,4,5,6,7,8,9,10,15], a,'LineWidth',2);
set(h1(1),'Color','#77AC30','LineStyle','-.')
set(h1(2),'Color','#4DBEEE','LineStyle','--')
% set(gca,'fontsize',12)
grid on
xlabel('Learning rate')
ylabel('Test accuracy (%)')
legend('Adadelta','Ours','fontsize',12)
h = subplot(2,1,2);
plot(log([1e-5,0.01]),[1.4,1.4], log([1e-5, 0.02]), [1.2, 1.2], log([1e-5,0.04]), [1,1],...
    log([1e-4, 0.2]), [0.8 0.8], log([5e-4, 0.2]), [0.6, 0.6],log([1e-3,0.8]),[0.4,0.4],...
    log([5e-3,9]),[0.2,0.2],log([1e-5,15]),[0,0],'LineWidth',2)
set(h,'xgrid','on')
xlabel('Learning rate (log)')
yticks([0:0.2:1.4])
yticklabels({'Ours','Adadelta','SGD','Momen.','Adagrad','Adamax','Adam','RMSprop'})
% ylim([-0.1,1.05])
% {Adacomp,Adadelta,SGD,Adagrad,Momentum,RMSprop}

%关于不同训练批大小数据，在表test_acc_batch中，学习率lr=0.01为固定值
test_acc_batch = [
    97.06, 98.22, 98.51, 98.79, 98.78, 98.76, 98.91, 98.95, 98.65, 98.99; % SGD, bz=16
    95.95, 97.47, 98.01, 98.32, 98.5, 98.61, 98.67, 98.48, 98.67, 98.74; % SGD, bz=32
    94.75, 94.81, 96.51, 97.39, 97.92, 97.97, 98.14, 98.08, 98.16, 98.42; % SGD, bz=64
    92.43, 94.38, 95.84, 96.7, 96.92, 97.05, 97.55, 97.61, 97.36, 97.52; % SGD, bz=128
    98.67, 98.31, 98.87, 98.85, 98.79, 98.78, 99.03, 98.92, 98.98, 98.73; %Momentum, bz=16
    98.46, 98.72, 98.86, 98.9, 98.99, 98.94, 98.96, 98.89, 98.9, 98.92; %Momentum, bz=32
    98.41, 98.68, 98.68, 98.86, 98.89, 98.98, 98.95, 98.9, 98.86, 98.85; %Momentum, bz=64
    98.24, 98.56, 98.71, 98.81, 98.74, 98.77, 98.8, 98.8, 98.76, 98.78; %Momentum, bz=128
    98.48, 98.79, 98.8, 98.89, 98.94, 98.96, 98.96, 99.0, 98.92, 98.95; %Adagrad, bz=16
    98.46, 98.72, 98.86, 98.9, 98.99, 98.94, 98.96, 98.89, 98.9, 98.92; %Adagrad, bz=32
    98.41, 98.68, 98.68, 98.86, 98.89, 98.98, 98.95, 98.9, 98.86, 98.85; %Adagrad, bz=64
    98.24, 98.56, 98.71, 98.81, 98.74, 98.77, 98.8, 98.8, 98.76, 98.78; %Adagrad, bz=128
    96.55, 96.11, 96.78, 97.36, 97.52, 97.54, 96.87, 97.54, 96.79, 96.58; %RMSprop, bz=16
    97.14, 96.92, 97.23, 97.33, 97.54, 97.65, 97.75, 97.68, 97.69, 96.99; %RMSprop, bz=32
    95.09, 95.78, 97.75, 98.03, 97.76, 98.11, 98.01, 98.17, 98.01, 98.82; %RMSprop, bz=64
    94.85, 93.01, 96.67, 97.39, 96.55, 97.11, 97.26, 97.84, 96.83, 97.7; %RMSprop, bz=128
    94.84, 96.82, 97.56, 97.88, 98.2, 98.28, 98.4, 98.31, 98.43, 98.55; %Adadelta, bz=16
    93.77, 96.11, 96.99, 97.61, 97.95, 98.04, 98.23, 98.16, 98.25, 98.47; %Adadelta, bz=32
    92.58, 94.41, 95.96, 96.75, 97.19, 97.62, 97.82, 97.88, 97.93, 98.24; %Adadelta, bz=64
    91.27, 93.1, 94.49, 95.39, 95.85, 96.59, 96.97, 97.04, 97.26, 97.55; %Adadelta, bz=128
    11.35, 11.35, 11.35, 10.28, 11.35, 10.1, 11.35, 11.35, 11.35, 11.35; % Adam, bz=16
    96.61, 96.99, 96.89, 97.17, 96.25, 95.75, 97.34, 97.0, 97.06, 97.25; % Adam, bz=32
    95.67, 95.93, 96.45, 97.04, 95.97, 97.1, 96.85, 97.45, 96.96, 97.4; % Adam, bz=64
    97.45, 97.98, 98.06, 98.28, 98.28, 98.29, 97.76, 98.22, 97.95, 97.91; % Adam, bz=128
    98.09, 98.21, 98.65, 98.79, 98.86, 98.8, 98.85, 98.98, 98.73, 99.07; % Adamax, bz=16
    98.12, 98.23, 98.69, 98.6, 98.73, 98.72, 98.88, 98.77, 98.86, 98.78; % Adamax, bz=32
    98.51, 98.66, 98.75, 98.85, 98.86, 98.77, 99.02, 98.91, 98.93, 98.67; % Adamax, bz=64
    98.41, 98.88, 98.85, 98.93, 98.92, 98.95, 98.96, 99.13, 98.93, 99.06; % Adamax, bz=128    
    97.71, 97.92, 97.92, 97.92, 97.92, 98.46, 98.6, 98.69, 98.85, 98.8; %Adacomp, bz=16
    97.15, 97.26, 97.28, 97.7, 98.18, 98.42, 98.56, 98.71, 98.64, 98.73; %Adacomp, bz=32
    97.62, 98.3, 98.41, 98.43, 98.41, 98.67, 98.79, 98.82, 98.81, 98.84; %Adacomp, bz=64
    95.04, 95.29, 97.53, 98.1, 98.32, 98.62, 98.57, 98.61, 98.54, 98.67; %Adacomp, bz=128     
    ];
figure(3)
subplot(2,4,1)
plot(test_acc_batch(1:4,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
title('SGD')
subplot(2,4,2)
plot(test_acc_batch(5:8,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Momentum')
subplot(2,4,3)
plot(test_acc_batch(9:12,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adagrad')
subplot(2,4,4)
legend('16','32','64','128','Location','southoutside','NumColumns',4)
plot(test_acc_batch(13:16,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('RMSprop')
subplot(2,4,5)
plot(test_acc_batch(17:20,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
title('Adadelta')
subplot(2,4,6)
plot(test_acc_batch(21:24,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adam')
subplot(2,4,7)
plot(test_acc_batch(25:28,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adamax')
subplot(2,4,8)
plot(test_acc_batch(29:end,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Ours')

%关于不同模型初始值的对比数据
test_acc_seed = [
    94.73, 95.02, 96.41, 97.53, 97.85, 98.08, 98.13, 98.08, 98.07, 98.38; %SGD, seed=1 
    93.49, 95.56, 96.99, 96.97, 98.02, 98.08, 97.99, 98.25, 98.34, 98.4; %SGD, seed=10
    92.63, 96.38, 96.87, 97.6, 97.93, 98.2, 97.5, 98.55, 98.38, 97.66; %SGD, seed=30
    94.37, 95.25, 97.0, 97.01, 97.68, 97.15, 97.63, 98.16, 97.86, 98.3; %SGD, seed=50
    98.25, 98.58, 98.67, 98.97, 98.97, 99.05, 99.0, 98.97, 98.88, 99.08; %Mommentum, seed=1 
    98.14, 98.89, 98.58, 98.79, 98.86, 98.91, 99.01, 99.02, 99.1, 99.02; %Mommentum, seed=10
    97.98, 98.78, 98.65, 98.8, 99.02, 98.82, 99.0, 98.98, 99.04, 99.1; %Mommentum, seed=30
        97.92, 98.59, 98.8, 99.01, 99.06, 98.83, 99.11, 99.01, 98.94, 98.96; %Mommentum, seed=50
    98.37, 98.7, 98.67, 98.92, 98.87, 98.97, 98.95, 98.87, 98.9, 98.83; %Adagrad, seed=1 
    98.15, 98.64, 98.71, 98.8, 98.73, 98.84, 98.84, 98.9, 98.95, 98.92; %Adagrad, seed=10
    98.51, 98.79, 98.8, 98.94, 99.02, 98.95, 98.99, 98.97, 98.98, 99.05; %Adagrad, seed=30
        98.3, 98.71, 98.58, 98.87, 98.99, 98.92, 98.82, 98.95, 98.96, 98.97; %Adagrad, seed=50
    95.9, 96.1, 97.2, 97.65, 97.53, 96.3, 97.12, 98.01, 96.31, 98.11; %RMSprop, seed=1 
    82.37, 96.01, 97.51, 97.19, 97.34, 97.05, 97.65, 98.12, 98.15, 98.08; %RMSprop, seed=10
    11.35, 11.35, 11.35, 11.35, 10.1, 11.35, 11.35, 11.35, 10.28, 11.35; %RMSprop, seed=30
        11.35, 11.35, 9.8, 11.35, 11.35, 11.35, 10.28, 11.35, 11.35, 10.09; %RMSprop, seed=50
    92.61, 94.43, 95.99, 96.73, 97.21, 97.61, 97.78, 97.89, 97.96, 98.23; %Adadelta, seed=1 
    92.17, 94.3, 95.5, 96.44, 97.13, 97.37, 97.77, 97.84, 97.86, 98.06; %Adadelta, seed=10
    91.58, 94.3, 95.42, 96.4, 96.97, 97.45, 97.66, 98.01, 98.13, 98.2; %Adadelta, seed=30
        92.53, 94.2, 95.5, 96.02, 96.78, 97.09, 97.49, 97.9, 98.08, 98.28; %Adadelta, seed=50
    95.67, 95.93, 96.45, 97.04, 95.97, 97.1, 96.85, 97.45, 96.96, 97.4; %Adam, seed=1 
    96.8, 97.64, 97.41, 97.7, 97.86, 97.35, 97.62, 97.58, 97.75, 97.94; %Adam, seed=10
    96.9, 96.94, 98.34, 98.18, 98.15, 97.76, 97.84, 98.17, 97.72, 97.67; %Adam, seed=30
        97.28, 97.68, 97.73, 97.28, 97.9, 97.98, 98.12, 97.97, 95.46, 97.91; %Adam, seed=50
    98.51, 98.66, 98.75, 98.85, 98.86, 98.77, 99.02, 98.91, 98.93, 98.67; %Adamax, seed=1 
    98.39, 98.42, 98.53, 98.82, 98.79, 98.97, 98.93, 98.79, 98.81, 98.7; %Adamax, seed=10
    97.83, 98.57, 98.85, 99.1, 98.8, 98.82, 98.85, 98.91, 98.75, 98.85; %Adamax, seed=30
        97.83, 98.28, 98.45, 98.69, 98.66, 98.46, 98.6, 98.56, 98.57, 98.81; %Adamax, seed=50
    97.62, 98.3, 98.41, 98.43, 98.41, 98.67, 98.79, 98.82, 98.81, 98.84; %Adacomp, seed=1 
    98.46, 98.48, 98.5, 98.59, 98.61, 98.71, 98.8, 98.86, 98.89, 99.02; %Adacomp, seed=10
    97.02, 97.66, 98.15, 98.4, 98.43, 98.62, 98.83, 98.96, 98.93, 98.94; %Adacomp, seed=30
    97.71, 97.8, 98.23, 98.29, 98.24, 98.6, 98.62, 98.79, 98.77, 98.86; %Adacomp, seed=50
    ];
%关于不同模型初始值的对比图示
figure(4)
subplot(2,4,1)
plot(test_acc_seed(1:4,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
title('SGD')
subplot(2,4,2)
plot(test_acc_seed(5:8,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Momentum')
subplot(2,4,3)
plot(test_acc_seed(9:12,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adagrad')
subplot(2,4,4)
plot(test_acc_seed(13:16,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('RMSprop')
subplot(2,4,5)
plot(test_acc_seed(17:20,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
legend('1','10','30','50','NumColumns',4)
grid on
ylim([94,100])
title('Adadelta')
subplot(2,4,6)
plot(test_acc_seed(21:24,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adam')
subplot(2,4,7)
plot(test_acc_seed(25:28,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Adamax')
subplot(2,4,8)
plot(test_acc_seed(29:end,:)', 'LineWidth',2)
set(gca, 'FontSize',12)
grid on
ylim([94,100])
yticklabels({})
title('Ours')

%Adacomp关于参量beta的变化 MNIST
acc_with_clip_001 = [[97.62, 98.3, 98.41, 98.43, 98.41, 98.57, 98.66, 98.69, 98.74, 98.71] % beta=0.6, lr=0.01
    [97.56, 98.02, 98.36, 98.36, 98.34, 98.5, 98.56, 98.64, 98.7, 98.72]  %beta=0.7
    [97.64, 97.89, 98.19, 98.22, 98.24, 98.34, 98.38, 98.42, 98.52, 98.53] % beta=0.8
    [97.79, 98.07, 98.08, 98.3, 98.36, 98.55, 98.57, 98.64, 98.69, 98.74] % beta=0.9
    [97.63, 97.84, 98.03, 98.05, 98.09, 98.23, 98.4, 98.47, 98.48, 98.62] % beta=1.0
    [97.22, 97.99, 98.13, 98.21, 98.26, 98.52, 98.56, 98.61, 98.69, 98.74]  % beta=1.5
    [97.67, 98.42, 98.44, 98.47, 98.49, 98.64, 98.72, 98.71, 98.78, 98.87] % beta=2.0
    [97.52, 98.17, 98.3, 98.17, 98.4, 98.52, 98.59, 98.64, 98.63, 98.75] % beta=2.5
    [98.05, 98.46, 98.55, 98.56, 98.56, 98.66, 98.7, 98.78, 98.84, 98.86] % beta=3.0
    [97.49, 98.18, 98.14, 98.22, 98.21, 98.43, 98.62, 98.6, 98.72, 98.75] % beta=3.5
    [97.71, 98.42, 98.53, 98.59, 98.51, 98.63, 98.63, 98.6, 98.72, 98.75] % beta=4.0
    [93.96, 97.16, 97.16, 97.22, 97.22, 97.53, 97.7, 97.86, 97.9, 98.16]  % beta=4.5
    [97.76, 98.27, 98.42, 98.46, 98.04, 98.39, 98.56, 98.5, 98.65, 98.72] % beta=5.0
    ];
acc_withoutclip_001 = [[97.82, 98.31, 98.49, 98.59, 98.58, 98.69, 98.74, 98.8, 98.84, 98.88] % beta=0.6, lr=0.01
    [97.64, 97.98, 98.2, 98.23, 98.25, 98.43, 98.51, 98.62, 98.59, 98.67]  %beta=0.7
    [97.9, 98.43, 98.6, 98.62, 98.68, 98.77, 98.76, 98.86, 98.77, 98.96] % beta=0.8
    [97.18, 98.39, 98.5, 98.51, 98.44, 98.65, 98.68, 98.69, 98.71, 98.76] % beta=0.9
    [97.1, 97.45, 97.61, 97.63, 97.67, 97.75, 97.89, 98.0, 98.11, 98.18] % beta=1.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]  % beta=1.5
    [97.67, 98.42, 98.44, 98.47, 98.49, 98.64, 98.72, 98.71, 98.78, 98.87]  % beta=2.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=2.5
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=3.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=3.5
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=4.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=4.5
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=5.0
    ];

acc_with_clip_01 = [[97.86, 98.35, 98.5, 98.53, 98.51, 98.68, 98.67, 98.78, 98.8, 98.87]  % beta=0.6, lr=0.1
    [97.93, 98.36, 98.46, 98.43, 98.52, 98.61, 98.7, 98.76, 98.82, 98.91]  %beta=0.7
    [97.47, 98.34, 98.52, 98.52, 98.57, 98.61, 98.68, 98.72, 98.71, 98.78] % beta=0.8
    [98.04, 98.51, 98.64, 98.66, 98.63, 98.76, 98.77, 98.84, 98.78, 98.85]  % beta=0.9
    [97.73, 98.27, 98.38, 98.4, 98.63, 98.68, 98.66, 98.81, 98.82, 98.88] % beta=1.0
    [97.93, 98.25, 98.46, 98.46, 98.46, 98.56, 98.66, 98.75, 98.72, 98.8]  % beta=1.5
    [97.75, 98.46, 98.59, 98.62, 98.6, 98.7, 98.74, 98.79, 98.8, 98.81] % beta=2.0
    [97.61, 98.11, 98.25, 98.39, 98.62, 98.8, 98.82, 98.82, 98.91, 98.85] % beta=2.5
    [97.97, 95.21, 97.83, 98.06, 98.3, 98.39, 98.49, 98.51, 98.57, 98.61] % beta=3.0
    [97.49, 98.18, 98.44, 98.45, 98.45, 98.6, 98.7, 98.76, 98.8, 98.84] % beta=3.5
    [97.65, 98.16, 98.35, 98.35, 98.37, 98.59, 98.69, 98.73, 98.73, 98.74] % beta=4.0
    [95.34, 96.83, 97.62, 97.6, 97.85, 98.11, 98.16, 98.23, 98.37, 98.36]  % beta=4.5
    [97.63, 97.89, 98.08, 98.08, 98.08, 98.21, 98.3, 98.33, 98.45, 98.53]  % beta=5.0
    ];
acc_withoutclip_01 = [[97.86, 98.35, 98.5, 98.53, 98.51, 98.68, 98.67, 98.78, 98.8, 98.87] % beta=0.6, lr=0.1
    [97.71, 98.38, 98.41, 98.46, 98.46, 98.63, 98.68, 98.76, 98.85, 98.83]   %beta=0.7
    [97.47, 98.34, 98.52, 98.52, 98.57, 98.61, 98.68, 98.72, 98.71, 98.78]  % beta=0.8
    [98.03, 98.61, 98.68, 98.7, 98.68, 98.8, 98.81, 98.9, 98.9, 98.97]  % beta=0.9
    [98.07, 98.39, 98.43, 98.48, 98.49, 98.54, 98.57, 98.57, 98.62, 98.7] % beta=1.0
    [98.0, 98.39, 98.45, 98.52, 98.52, 98.66, 98.76, 98.82, 98.82, 98.93]  % beta=1.5
    [96.85, 97.88, 98.06, 97.97, 98.03, 98.15, 98.24, 98.29, 98.3, 98.37]  % beta=2.0
    [98.19, 98.49, 98.52, 98.55, 98.62, 98.8, 98.89, 98.91, 98.94, 98.9]  % beta=2.5
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=3.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=3.5
    [96.61, 98.33, 98.46, 98.48, 98.61, 98.83, 98.92, 98.97, 98.95, 99.08]  % beta=4.0
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=4.5
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] % beta=5.0
    ];

acc_with_clip_1 = [[97.76, 98.34, 98.54, 98.63, 98.6, 98.6, 98.7, 98.7, 98.72, 98.76] % beta=0.6, lr=1
    [97.28, 98.27, 98.48, 98.58, 98.65, 98.71, 98.82, 98.82, 98.8, 98.81]  %beta=0.7
    [97.68, 98.03, 98.25, 98.5, 98.45, 98.55, 98.62, 98.61, 98.64, 98.7] % beta=0.8
    [97.68, 98.32, 98.54, 98.53, 98.56, 98.64, 98.64, 98.68, 98.76, 98.64] % beta=0.9
    [97.42, 97.35, 98.13, 98.16, 98.19, 98.24, 98.37, 98.38, 98.5, 98.4] % beta=1.0
    [97.66, 98.04, 98.29, 98.45, 98.44, 98.49, 98.65, 98.68, 98.69, 98.82]   % beta=1.5
    [97.21, 97.83, 98.02, 98.02, 98.04, 98.18, 98.23, 98.38, 98.34, 98.49] % beta=2.0
    [97.01, 97.15, 97.76, 97.69, 98.03, 98.26, 98.41, 98.49, 98.48, 98.5] % beta=2.5
    [96.68, 97.91, 98.08, 98.12, 98.17, 98.21, 98.32, 98.26, 98.33, 98.33] % beta=3.0
    [96.56, 96.81, 96.99, 97.35, 97.64, 97.82, 97.95, 98.14, 98.19, 98.26] % beta=3.5
    [97.4, 98.25, 98.43, 98.54, 98.63, 98.71, 98.71, 98.75, 98.81, 98.83] % beta=4.0
    [96.36, 97.53, 97.88, 98.37, 98.39, 98.43, 98.5, 98.55, 98.58, 98.57]  % beta=4.5
    [10.32, 95.59, 97.32, 97.43, 96.81, 97.8, 98.35, 98.38, 98.56, 98.71] % beta=5.0
    ];
acc_withoutclip_1 = [[97.44, 97.71, 98.17, 98.2, 98.21, 98.33, 98.41, 98.35, 98.48, 98.53]  % beta=0.6, lr=1
    [97.83, 97.97, 98.35, 98.49, 98.55, 98.56, 98.62, 98.67, 98.72, 98.77]   %beta=0.7
    [96.4, 97.3, 97.77, 97.79, 97.78, 97.91, 98.07, 98.02, 98.05, 98.08] % beta=0.8
    [97.47, 97.81, 97.96, 98.29, 98.36, 98.51, 98.61, 98.67, 98.7, 98.71]  % beta=0.9
    [97.38, 97.9, 98.03, 98.07, 98.11, 98.22, 98.32, 98.34, 98.42, 98.34]  % beta=1.0
    [97.89, 98.27, 98.48, 98.48, 98.49, 98.53, 98.55, 98.62, 98.66, 98.65]   % beta=1.5
    [97.21, 97.83, 98.02, 98.02, 98.04, 98.18, 98.23, 98.38, 98.34, 98.49]   % beta=2.0
    [97.01, 97.15, 97.76, 97.69, 98.03, 98.26, 98.41, 98.49, 98.48, 98.5]   % beta=2.5
    [92.77, 93.97, 94.21, 94.2, 94.35, 95.22, 95.76, 96.18, 96.42, 96.69] % beta=3.0
    [96.56, 96.81, 96.99, 97.35, 97.64, 97.82, 97.95, 98.14, 98.19, 98.26]  % beta=3.5
    [11.34, 11.34, 11.34, 11.34, 11.34, 11.34, 11.34, 11.34, 11.34, 11.34]  % beta=4.0
    [96.36, 97.53, 97.88, 98.37, 98.39, 98.43, 98.5, 98.55, 98.58, 98.57]  % beta=4.5
    [11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35] % beta=5.0
    ];
figure(5)
accuracy = [acc_withoutclip_001(:,end),acc_with_clip_001(:,end),...
    acc_withoutclip_01(:,end),acc_with_clip_01(:,end),...
    acc_withoutclip_1(:,end),acc_with_clip_1(:,end)];

beta = [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
% method = {'Unclipped', 'Clipped', 'Unclipped', 'Clipped', 'Unclipped', 'Clipped'};
method = {'LR=0.01','LR=0.01(c)','LR=0.1','LR=0.1(c)','LR=1','LR=1(c)'};
heatmap(method, beta, accuracy, 'FontSize',12, 'FontName','Times New Roman',...
    'MissingDataColor',[0.941176470588235 0.941176470588235 0.941176470588235])
xlabel("Learning rate (LR) with or without clipping gradients")
ylabel("\beta")
title("Sensitivity of \beta in Adacomp")





