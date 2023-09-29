import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('results.csv')

# Crear la nueva columna "winner" utilizando condiciones
df['winner'] = df.apply(lambda row: row['home_team'] if row['home_score'] > row['away_score'] else 
                        (row['away_team'] if row['away_score'] > row['home_score'] else 'DRAW'), axis=1)

# Guardar el DataFrame actualizado en un nuevo archivo CSV
df.to_csv('results_with_winner.csv', index=False)