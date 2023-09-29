import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict

def train_model(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Definir la red neuronal
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Dos neuronas de salida para predecir la tasa de goles de ambos equipos

    # Compilar la red neuronal
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Entrenar la red neuronal
    model.fit(X_train, y_train, epochs=150, batch_size=10)

    return model

def simulate_game(model, X, df):
    # Usar la red neuronal para predecir la tasa de goles para cada equipo en cada partido
    predicted_rates = model.predict(X)

    # Asegurarse de que las tasas de goles sean al menos cero
    predicted_rates = np.maximum(predicted_rates, 0)

    # Usar la distribución de Poisson para generar un número aleatorio de goles para cada equipo en cada partido
    predicted_goals = np.random.poisson(predicted_rates)

    # Crear un DataFrame para almacenar los resultados de los partidos
    results = pd.DataFrame(predicted_goals, columns=['goals1', 'goals2'])
    results['team1'] = df['home_team']
    results['team2'] = df['away_team']

    # Determinar el ganador de cada partido
    results['winner'] = results.apply(lambda row: row['team1'] if row['goals1'] > row['goals2'] else 
                                    (row['team2'] if row['goals2'] > row['goals1'] else 'DRAW'), axis=1)

    # Aplicar el sistema de puntuación
    results['points1'] = results.apply(lambda row: 3 if row['winner'] == row['team1'] else 
                                    (1 if row['winner'] == 'DRAW' else 0), axis=1)
    results['points2'] = results.apply(lambda row: 3 if row['winner'] == row['team2'] else 
                                    (1 if row['winner'] == 'DRAW' else 0), axis=1)

    # Sumar los puntos de cada equipo
    points = results.groupby('team1')['points1'].sum() + results.groupby('team2')['points2'].sum()

    # El equipo con más puntos es el ganador
    winner = points.idxmax()

    return winner

# Cargar los datos históricos
df = pd.read_csv('results_with_winner.csv')

# Preparar los datos para el entrenamiento de la red neuronal
X = df[['home_team', 'away_team']].values
y = df[['home_score', 'away_score']].values

# Normalizar los datos en y
scaler = MinMaxScaler()
y = scaler.fit_transform(y)

# Codificar las etiquetas de los equipos
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])

X = X.astype(float)

# Entrenar el modelo una vez al principio
model = train_model(X, y)

# Diccionario para almacenar los ganadores
winners = defaultdict(int)

# Realizar 10000 iteraciones
for _ in range(10000):
    winner = simulate_game(model, X, df)
    winners[winner] += 1

# Calcular la media
mean_wins = {team: wins / 10000 for team, wins in winners.items()}

# Imprimir las estadísticas
for team, mean in mean_wins.items():
    print(f'El equipo {team} ganó en promedio {mean} veces.')

# Generar la gráfica
teams = list(mean_wins.keys())
means = list(mean_wins.values())

plt.bar(teams, means)
plt.xlabel('Equipos')
plt.ylabel('Victorias promedio')
plt.title('Victorias promedio de los equipos en 10000 simulaciones')
plt.show()