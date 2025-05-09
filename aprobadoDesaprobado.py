import pandas as pd
import numpy as np


data = {
    'Horas_Estudio': [2, 5, 8, 1, 6, 3, 7, 4, 9, 10],
    'Conocimiento_Previo': [3, 6, 8, 2, 7, 4, 9, 5, 9, 10],
    'Asistencia': [60, 75, 90, 50, 85, 65, 95, 70, 98, 100],
    'Promedio_Tareas': [6, 7, 9, 5, 8, 6, 9, 7, 10, 10],
    'Estudio_Grupo': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'Resultado': [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]  # 0 = No Aprobado, 1 = Aprobado
}


df = pd.DataFrame(data)


def predecir_aprobado(fila):
    score = 0
    score += fila['Horas_Estudio'] * 0.2
    score += fila['Conocimiento_Previo'] * 0.3
    score += (fila['Asistencia'] / 100) * 0.2
    score += fila['Promedio_Tareas'] * 0.2
    score += fila['Estudio_Grupo'] * 0.1
    return 1 if score > 5.5 else 0


df['Prediccion'] = df.apply(predecir_aprobado, axis=1)


print(df[['Resultado', 'Prediccion']])
print("\nPrecisión:", np.mean(df['Resultado'] == df['Prediccion']))
