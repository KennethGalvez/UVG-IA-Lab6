import csv
import pandas as pd

# Cargar el archivo csv en un DataFrame
df = pd.read_csv('high_diamond_ranked_10min.csv')

# Imprimir las primeras dos filas
print(df.head(2))

with open('high_diamond_ranked_10min.csv', 'r') as file:
    reader = csv.reader(file)
    first_row = next(reader)
    row_length = len(first_row)
    print(f"La longitud de la primera fila es {row_length}")
    
    for i, row in enumerate(reader):
        if len(row) != row_length:
            print(f"Error en la fila {i + 2}: la longitud de la fila es diferente a la de la primera fila")
        if not row:
            print(f"Error en la fila {i + 2}: la fila está vacía")
        if '' in row:
            print(f"Error en la fila {i + 2}: hay valores faltantes en la fila")


