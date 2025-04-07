import pandas as pd
import torch
import torch.nn as nn

# Значение n из ЭИОС
n = 17

# Условие для выбора задачи
if n % 2 == 1:
    print('Решите задачу классификации покупателей на классы *купит* - *не купит* (3й столбец) по признакам возраст и доход.')
    
    # Загрузка данных
    df = pd.read_csv('dataset_simple.csv')
    X = df.iloc[:, 0:2].values  # Первые два столбца: возраст и доход
    y = df.iloc[:, 2].values.reshape(-1, 1)  # Третий столбец: класс ("купит" или "не купит")

    # Преобразование меток классов в числовые значения (1 - "купит", 0 - "не купит")
    y = (y == "купит").astype(float)

    # Нормализация данных 
    mean_X = X.mean(axis=0)  # Среднее значение по каждому признаку
    std_X = X.std(axis=0)    # Стандартное отклонение по каждому признаку
    X = (X - mean_X) / std_X  # Стандартизация данных
    X = torch.Tensor(X)       # Преобразование в тензор PyTorch
    y = torch.Tensor(y)       # Преобразование меток в тензор PyTorch

    # Определение архитектуры нейронной сети для классификации
    class NNet(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super(NNet, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, out_size),
                nn.Sigmoid()
            )
        
        def forward(self, X):
            return self.layers(X)

    # Параметры сети
    input_size = X.shape[1]
    hidden_size = 5  # Число нейронов в скрытом слое
    output_size = 1  # Один выходной нейрон для бинарной классификации

    # Создание экземпляра сети
    net = NNet(input_size, hidden_size, output_size)

    # Функция потерь и оптимизатор
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Обучение модели
    epochs = 100
    for epoch in range(epochs):
        pred = net(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Ошибка на эпохе {epoch + 1}: {loss.item()}')

    # Оценка модели
    with torch.no_grad():
        pred = net(X)
        pred_labels = (pred >= 0.5).float()
        accuracy = (pred_labels == y).float().mean()
        print(f'Точность модели: {accuracy.item() * 100:.2f}%')

else:
    print('Решите задачу предсказания дохода по возрасту.')

    # Загрузка данных
    df = pd.read_csv('dataset_simple.csv')
    X = df.iloc[:, 0].values.reshape(-1, 1)  # Первый столбец: возраст
    y = df.iloc[:, 1].values.reshape(-1, 1)  # Второй столбец: доход

    # Нормализация данных 
    mean_X = X.mean(axis=0)  # Среднее значение по каждому признаку
    std_X = X.std(axis=0)    # Стандартное отклонение по каждому признаку
    X = (X - mean_X) / std_X  # Стандартизация данных
    X = torch.Tensor(X)       # Преобразование в тензор PyTorch
    y = torch.Tensor(y)       # Преобразование меток в тензор PyTorch

    # Определение архитектуры нейронной сети для регрессии
    class NNetRegression(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super(NNetRegression, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size)
            )
        
        def forward(self, X):
            return self.layers(X)

    # Параметры сети
    input_size = X.shape[1]
    hidden_size = 5  # Число нейронов в скрытом слое
    output_size = 1  # Один выходной нейрон для регрессии

    # Создание экземпляра сети
    net = NNetRegression(input_size, hidden_size, output_size)

    # Функция потерь и оптимизатор
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Обучение модели
    epochs = 100
    for epoch in range(epochs):
        pred = net(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Ошибка на эпохе {epoch + 1}: {loss.item()}')

    # Оценка модели
    with torch.no_grad():
        pred = net(X)
        mae = torch.mean(torch.abs(y - pred)).item()
        print(f'Ошибка (MAE): {mae:.2f}')