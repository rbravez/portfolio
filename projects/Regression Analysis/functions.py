import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from itertools import product
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from itertools import chain
import scipy.stats as stats
import warnings
from functools import reduce
import math

warnings.filterwarnings("ignore")

def curva_roc(x_test_modelo, y_test, modelo):
    """
    Calcula la curva ROC y muestra la gráfica.

    Parameters:
        x_test_modelo (array-like): Conjunto de datos de prueba.
        y_test (array-like): Etiquetas verdaderas de prueba.
        modelo (dict): Diccionario que contiene el modelo ajustado.

    Returns:
        float: Área bajo la curva ROC (AUC).
    """
    # Calcula las probabilidades de la clase positiva
    y_prob = modelo['Modelo'].predict_proba(x_test_modelo)[:, 1]
    
    # Calcula la tasa de falsos positivos (FPR) y la tasa de verdaderos positivos (TPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Grafica la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()
    
    # Calcula el Área bajo la curva ROC (AUC)
    AUC = roc_auc_score(y_test, y_prob)
    print('Área bajo la curva ROC =', AUC)
    
    return AUC


def analizar_variables_categoricas(datos):
    """
    Analiza variables categóricas en un DataFrame.

    Args:
        datos (DataFrame): El DataFrame que contiene los datos.

    Returns:
        dict: Un diccionario donde aparecen las diferentes categorias, sus frecuencias
        absolutas y relativas.
    """
    # Inicializar un diccionario para almacenar los resultados
    resultados = {}
    
    # Genera una lista con los nombres de las variables.
    variables = list(datos.columns) 
    
    # Seleccionar las columnas numéricas en el DataFrame
    numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

    # Seleccionar las columnas categóricas en el DataFrame
    categoricas = [variable for variable in variables if variable not in numericas]
    
    # Iterar a través de las variables categóricas
    for categoria in categoricas:
        # Verificar si la variable categórica existe en el DataFrame
        if categoria in datos.columns:
            # Crear un DataFrame de resumen para la variable categórica
            resumen = pd.DataFrame({
                'n': datos[categoria].value_counts(),             # Conteo de frecuencias
                '%': datos[categoria].value_counts(normalize=True)  # Porcentaje de frecuencias
            })
            resultados[categoria] = resumen  # Almacenar el resumen en el diccionario
        else:
            # Si la variable no existe en los datos, almacenar None en el diccionario
            resultados[categoria] = None
    
    return resultados

def cuentaDistintos(datos):
    """
    Cuenta valores distintos en cada variable numerica de un DataFrame.

    Args:
        datos (DataFrame): El DataFrame que contiene los datos.

    Returns:
        Dataframe: Un DataFrame con las variables y valores distintos en cada una de ellas
    """
    # Seleccionar las columnas numéricas en el DataFrame
    numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64'])
    
    # Calcular la cantidad de valores distintos en cada columna numérica
    resultados = numericas.apply(lambda x: len(x.unique()))
    
    # Crear un DataFrame con los resultados
    resultado = pd.DataFrame({'Columna': resultados.index, 'Distintos': resultados.values})
    
    return resultado

def frec_variables_num(datos, NumCat):
    """
    Calcula las frecuencias de los diferentes valores de variables numericas (tratadas como categóricas).
    Args:
        datos: DataFrame de datos.
        NumCat: Lista de nombres de variables númericas a analizar.
        :return: Un diccionario donde las claves son los nombres de las variables numericas y los valores son DataFrames
             con el resumen de frecuencias y porcentajes.
    """
    resultados = {}

    for categoria in NumCat:
        # Verificar si la variable categórica existe en el DataFrame
        if categoria in datos.columns:
            # Crear un DataFrame de resumen para la variable categórica
            resumen = pd.DataFrame({
                'n': datos[categoria].value_counts(),             # Conteo de frecuencias
                '%': datos[categoria].value_counts(normalize=True)  # Porcentaje de frecuencias
            })
            resultados[categoria] = resumen  # Almacenar el resumen en el diccionario
        else:
            # Si la variable no existe en los datos, almacenar None en el diccionario
            resultados[categoria] = None
    
    return resultados



def atipicosAmissing(varaux):
    """
    Esta función identifica valores atípicos en una serie de datos y los reemplaza por NaN.
    
    Datos de entrada:
    - varaux: Serie de datos en la que se buscarán valores atípicos.
    
    Datos de salida:
    - Una nueva serie de datos con valores atípicos reemplazados por NaN.
    - El número de valores atípicos identificados.
    """
    
    # Verifica si la distribución de los datos es simétrica o asimétrica
    if abs(varaux.skew()) < 1:
        # Si es simétrica, calcula los valores atípicos basados en la desviación estándar
        criterio1 = abs((varaux - varaux.mean()) / varaux.std()) > 3
    else:
        # Si es asimétrica, calcula la Desviación Absoluta de la Mediana (MAD) y los valores atípicos
        mad = sm.robust.mad(varaux, axis=0)
        criterio1 = abs((varaux - varaux.median()) / mad) > 8
    
    # Calcula los cuartiles 1 (Q1) y 3 (Q3) para determinar el rango intercuartílico (H)
    qnt = varaux.quantile([0.25, 0.75]).dropna()
    Q1 = qnt.iloc[0]
    Q3 = qnt.iloc[1]
    H = 3 * (Q3 - Q1)
    
    # Identifica valores atípicos que están fuera del rango intercuartílico
    criterio2 = (varaux < (Q1 - H)) | (varaux > (Q3 + H))
    
    # Crea una copia de la serie original y reemplaza los valores atípicos por NaN
    var = varaux.copy()
    var[criterio1 & criterio2] = np.nan
    
    # Retorna la serie con valores atípicos reemplazados y el número de valores atípicos identificados
    return [var, sum(criterio1 & criterio2)]


def patron_perdidos(datos_input):
    """
    Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.

    Args:
        datos_input (DataFrame): El conjunto de datos de entrada.

    """
    # Calculo una matriz de correlación de los valores ausentes en las columnas con al menos un missing
    correlation_matrix = datos_input[datos_input.columns[datos_input.isna().sum() > 0]].isna().corr()
    
    # Creo una máscara para ocultar la mitad superior de la matriz (simetría)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Configuro el tamaño de la figura y el tamaño de la fuente en el gráfico
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    
    # Genero un mapa de calor (heatmap) de la matriz de correlación de valores ausentes
    # 'annot=True' muestra los valores dentro de las celdas
    # 'cmap='coolwarm'' establece la paleta de colores del mapa de calor
    # 'fmt=".2f"' formatea los valores como números de punto flotante con dos decimales
    # 'cbar=False' oculta la barra de color (escala) en el lado derecho
    # 'mask=mask' aplica la máscara para ocultar la mitad superior de la matriz
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False, mask=mask)
    
    # Establezco el título del gráfico
    plt.title("Matriz de correlación de valores ausentes")
    plt.show()

def ImputacionCuant(var, tipo):
    """
    Esta función realiza la imputación de valores faltantes en una variable cuantitativa.

    Datos de entrada:
    - var: Serie de datos cuantitativos con valores faltantes a imputar.
    - tipo: Tipo de imputación ('media', 'mediana' o 'aleatorio').

    Datos de salida:
    - Una nueva serie con valores faltantes imputados.
    """

    # Realiza una copia de la variable para evitar modificar la original
    vv = var.copy()

    if tipo == 'media':
        # Imputa los valores faltantes con la media de la variable
        vv[np.isnan(vv)] = round(np.nanmean(vv), 4)
    elif tipo == 'mediana':
        # Imputa los valores faltantes con la mediana de la variable
        vv[np.isnan(vv)] = round(np.nanmedian(vv), 4)
    elif tipo == 'aleatorio':
        # Imputa los valores faltantes de manera aleatoria basada en la distribución de valores existentes
        x = vv[~np.isnan(vv)]
        frec = x.value_counts(normalize=True).reset_index()
        frec.columns = ['Valor', 'Frec']
        frec = frec.sort_values(by='Valor')
        frec['FrecAcum'] = frec['Frec'].cumsum()
        random_values = np.random.uniform(min(frec['FrecAcum']), 1, np.sum(np.isnan(vv)))
        imputed_values = list(map(lambda x: list(frec['Valor'][frec['FrecAcum'] <= x])[-1], random_values))
        vv[np.isnan(vv)] = [round(x, 4) for x in imputed_values]

    return vv

def ImputacionCuali(var, tipo):
    """
    Esta función realiza la imputación de valores faltantes en una variable cualitativa.

    Datos de entrada:
    - var: Serie de datos cualitativos con valores faltantes a imputar.
    - tipo: Tipo de imputación ('moda' o 'aleatorio').

    Datos de salida:
    - Una nueva serie con valores faltantes imputados.
    """

    # Realiza una copia de la variable para evitar modificar la original
    vv = var.copy()

    if tipo == 'moda':
        # Imputa los valores faltantes con la moda (valor más frecuente)
        frecuencias = vv[~vv.isna()].value_counts()
        moda = frecuencias.index[np.argmax(frecuencias)]
        vv[vv.isna()] = moda
    elif tipo == 'aleatorio':
        # Imputa los valores faltantes de manera aleatoria a partir de valores no faltantes
        vv[vv.isna()] = np.random.choice(vv[~vv.isna()], size=np.sum(vv.isna()), replace=True)

    return vv


def Vcramer(v, target):
    """
    Calcula el coeficiente V de Cramer entre dos variables. Si alguna de ellas es continua, la discretiza.

    Datos de entrada:
    - v: Serie de datos categóricos o cuantitativos.
    - target: Serie de datos categóricos o cuantitativos.

    Datos de salida:
    - Coeficiente V de Cramer que mide la asociación entre las dos variables.
    """

    if v.dtype == 'float64' or v.dtype == 'int64':

        # Si v es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(v.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        v = pd.cut(v, bins=p)
        v = v.fillna(v.min())

    if target.dtype == 'float64' or target.dtype == 'int64':

        # Si target es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(target.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        target = pd.cut(target, bins=p)
        target = target.fillna(target.min())

    v = v.reset_index(drop=True)
    target = target.reset_index(drop=True)

    # Calcula una tabla de contingencia entre v y target
    tabla_cruzada = pd.crosstab(v, target)

    # Calcula el chi-cuadrado y el coeficiente V de Cramer
    chi2 = chi2_contingency(tabla_cruzada)[0]
    n = tabla_cruzada.sum().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla_cruzada.shape) - 1)))

    return v_cramer

    
def graficoVcramer(matriz, target):
    """
    Genera un gráfico de barras horizontales que muestra el coeficiente V de Cramer entre cada columna de matriz y la variable target.

    Datos de entrada:
    - matriz: DataFrame con las variables a comparar.
    - target: Serie de la variable objetivo (categórica).

    Datos de salida:
    - Gráfico de barras horizontales que muestra el coeficiente V de Cramer para cada variable.
    """

    # Calcula el coeficiente V de Cramer para cada columna de matriz y target
    salidaVcramer = {x: Vcramer(matriz[x], target) for x in matriz.columns}

    # Ordena los resultados en orden descendente por el coeficiente V de Cramer
    sorted_data = dict(sorted(salidaVcramer.items(), key=lambda item: item[1], reverse=True))

    # Crea el gráfico de barras horizontales
    plt.figure(figsize=(10, 6))
    plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color='skyblue')
    plt.xlabel('V de Cramer')
    plt.show()

    
def mosaico_targetbinaria(var, target, nombre_eje):
    """
    Genera un gráfico de mosaico (mosaic plot) que muestra la relación entre una variable 'var' y una variable objetivo binaria 'target'.

    Parámetros:
    - var: Variable explicativa cualitativa.
    - target: Variable binario (0 o 1) que representa la variable objetivo.
    - nombre_eje: El nombre de la variable 'var' para etiquetar el eje x en el gráfico.

    Salida:
    - Muestra un gráfico de mosaico que representa la relación entre 'var' y 'target'.
    """

    # Crear un DataFrame que contiene 'var' y 'target'
    df = pd.DataFrame({nombre_eje: var, 'target': target})

    # Calcular una tabla de frecuencia cruzada y normalizarla por filas
    tabla_frecuencia = df.groupby([nombre_eje, 'target']).size().unstack(fill_value=0)
    tabla_frecuencia = tabla_frecuencia.div(tabla_frecuencia.sum(axis=1), axis=0)

    # Definir colores para las barras del gráfico
    colores = ['#FF6666', '#6699FF']

    # Crear un gráfico de barras apiladas (mosaic plot)
    ax = tabla_frecuencia.plot(kind='bar', stacked=True, color=colores)

    # Establecer etiquetas para los ejes x e y
    plt.xlabel(nombre_eje)  # Eje x representa 'var'
    plt.ylabel('Frecuencia Target')  # Eje y representa la frecuencia de 'target'

    # Rotar las etiquetas del eje x para mejorar la legibilidad
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Mostrar el gráfico de mosaico
    plt.show()
    
   
    
    
def boxplot_targetbinaria(var, target, nombre_ejeX, nombre_ejeY):
    """
    Genera un boxplot para una variable 'var' en función de una variable objetivo binaria 'target'.

    Parámetros:
    - var: Variable explicativa continua.
    - target: Variable binario (0 o 1).
    - nombre_ejeX: El nombre del eje x (etiqueta) en el gráfico.
    - nombre_ejeY: El nombre del eje y (etiqueta) en el gráfico.

    Salida:
    - Muestra un gráfico de caja ('boxplot') que compara la variable 'var' para cada valor de 'target'.
    """

    # Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
    plt.figure(figsize=(8, 6))

    # Crear un boxplot donde 'x' representa los valores de 'target' e 'y' representa 'var'
    sns.boxplot(x=target, y=var)

    # Establecer etiquetas para los ejes x e y
    plt.xlabel(nombre_ejeX)  # Eje x representa la variable categorica
    plt.ylabel(nombre_ejeY)  # Eje y representa la variable continua

    # Mostrar el gráfico
    plt.show()    

    
def hist_targetbinaria(var, target, nombre_eje):
    """
    Genera un histograma de densidad para una variable 'var' en función de una variable objetivo binaria 'target'.

    Parámetros:
    - var: Variable explicativa continua.
    - target: Variable binario (0 o 1) que representa la variable objetivo.
    - nombre_eje: El nombre del eje x (etiqueta) en el gráfico.

    Salida:
    - Muestra un gráfico de densidad con dos curvas para 'target 0' y 'target 1'.
    """

    # Convertir la variable objetivo 'target' en una lista de números enteros (0 o 1)
    target_num = [int(x) for x in target]

    # Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
    plt.figure(figsize=(8, 6))

    # Crear un gráfico de densidad (Kernel Density Estimation) para 'var' cuando 'target' es igual a 0
    sns.kdeplot(var[[x == 0 for x in target_num]], label='target 0', fill=True)

    # Crear un gráfico de densidad (Kernel Density Estimation) para 'var' cuando 'target' es igual a 1
    sns.kdeplot(var[[x == 1 for x in target_num]], label='target 1', fill=True)

    # Establecer etiquetas para los ejes x e y
    plt.xlabel(nombre_eje)
    plt.ylabel('Density')

    # Mostrar una leyenda en el gráfico para identificar las curvas
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def hist_target_categorica(varCont, varCat, nombre_ejeX, nombre_ejeY):
    """
    Genera un histograma de densidad para una variable 'varCont' en función de una variable categórica 'varCat'.

    Parámetros:
    - varCont: Variable explicativa continua.
    - varCat: Variable categórica.
    - nombre_eje: El nombre del eje x (etiqueta) en el gráfico.

    Salida:
    - Muestra un gráfico de densidad con curvas para cada valor único de 'varCat'.
    """

    # Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
    plt.figure(figsize=(8, 6))

    # Obtener los valores únicos de la variable 'target'
    unique_values = varCat.unique()

    # Iterar sobre los valores únicos de 'target' y trazar las curvas de densidad correspondientes
    for value in unique_values:
        # Crear un gráfico de densidad (Kernel Density Estimation) para 'varCont' para cada valor de 'varCat'
        sns.kdeplot(varCont[varCat == value], label=nombre_ejeY+f' {value}', fill=True)

    # Establecer etiquetas para los ejes x e y
    plt.xlabel(nombre_ejeX)
    plt.ylabel('Density')

    # Mostrar una leyenda en el gráfico para identificar las curvas
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def bestTransfCorr(vv, target):
    """
    Function searches for the best transformation between a cuantiative variable that maximizes the correlation with the target variable
    Datos de entrada:
    - vv: Cuantiative data that we want to transform.
    - target: Target Variable.

    Datos de salida:
    - List that contains the name of the best transformation and the corresponding series.
    """

    # Normalization using StandardScaler
    vv = StandardScaler().fit_transform([[x] for x in list(vv)])

    # Values > 0
    vv = vv + abs(np.min(vv)) * 1.0001
    vv = [x[0] for x in vv]

    # Create a Dataframe
    posibles_transf = pd.DataFrame({
        'x': vv,
        'logx': np.log(vv),
        'expx': np.exp(vv),
        'sqrx': [x**2 for x in vv],
        'sqrtx': np.sqrt(vv),
        'cuartax': [x**4 for x in vv],
        'raiz4': [x**(1/4) for x in vv]
    })

    # Correlation between variablles
    cor_values = posibles_transf.apply(
        lambda col: np.abs(np.corrcoef(target, col, rowvar=False, ddof=0)[0, 1]),
        axis=0
    )

    # Best transformation
    max_corr_idx = cor_values.idxmax()

    return [max_corr_idx, posibles_transf[max_corr_idx]]
  
# Busca la transformacion de variables input de intervalo que maximiza la V de Cramer con la objetivo binaria

def mejorTransfVcramer(vv, target):
    """
    This function searches for the best transformation for a cuantitative variable that maximizes the coeficient of the Cramer V with a categorical target variable.

    Datos de entrada:
    - vv: Serie of cuantative data wished to transform.
    - target: Serie of categorical target variable for which we want to maximize V of Cramer.

    Datos de salida:
    - List that contains the name of the best transformation and the transformed series
    """

    # Normalization using StandardScaler
    vv = StandardScaler().fit_transform([[x] for x in list(vv)])

    # Positive values
    vv = vv + abs(np.min(vv)) * 1.0001
    vv = [x[0] for x in vv]

    # Creates a dataframe with possible transformations of the variable.
    posibles_transf = pd.DataFrame({
        'x': vv,
        'logx': np.log(vv),
        'expx': np.exp(vv),
        'sqrx': [x**2 for x in vv],
        'sqrtx': np.sqrt(vv),
        'cuartax': [x**4 for x in vv],
        'raiz4': [x**(1/4) for x in vv]
    })

    # Calculation of V of Cramer
    cor_values = posibles_transf.apply(
        lambda col: Vcramer(col, target),
        axis=0
    )

    # Find the best correlation
    max_corr_idx = cor_values.idxmax()

    return [max_corr_idx, posibles_transf[max_corr_idx]]


def Transf_Auto(matriz, target):
    """
    This functions does the automatic transformations for the columns of a dataframe in order to maximize the correlation or of the coeficient V of Cramer
    (if the target is categorical) with a target variable

    Input:
    - matriz: Dataframe with the variables to transform
    - target: Serie of target variable (numerical or categorical)

    Output:
    - Dataframe with the best transformations applied to the columns
    """

    if target.dtype in ['int64', 'float64']:
        # Usage of previous function if the variable is numerical
        aux = matriz.apply(
            lambda col: mejorTransfCorr(col, target),
            axis=0
        )
    else:
        # If the variable is categorical then we use the Cramer's V function
        aux = matriz.apply(
            lambda col: mejorTransfVcramer(col, target),
            axis=0
        )

    # Ouputs the best transformations and the corresponding series
    aux2 = aux.apply(lambda col: col[1], axis=0)
    aux = aux.apply(lambda col: col[0], axis=0)

    # Renames the columns with the name of the best transformation
    aux2.columns = [aux[x] + aux2.columns[x] for x in range(len(aux2.columns))]

    # Assigns the name indices to the original matrix
    aux2.index = matriz.index

    return aux2

def crear_data_modelo(datos, var_cont, var_categ, var_interac=[]):
    """
    Function to prepare the data for theit usage in the model, including numerical, categorical and interaction of the variables.

    Parameters:
    datos (DataFrame): Original dataframe
    var_cont (lista): List of continuous variables.
    var_categ (lista): List of categorical variables. 
    var_interac (lista, opcional): List of pair of variables for the interaction (by defect is a empty list).

    Returns:
    datos_v: Dataframe prepared with categorical variables and their interactions calculated
    """

    # Verification that there is data
    if len(var_cont + var_categ + var_interac) == 0:
        return datos[[]]

    # Selects variables continuous and categorical
    datos_v = datos[var_cont + var_categ]

    # Encodes categorical variables if they exists.
    if len(var_categ) > 0:
        datos_v = pd.get_dummies(datos_v, columns=var_categ, drop_first=True)

    # Calculates one list of all of the variables of entry.
    variables_total = list(set(var_cont + var_categ + list(chain.from_iterable(var_interac))))

    # Identifies the new categorical variables.
    var_categ_new = datos[variables_total].select_dtypes(include=['object', 'category']).columns.tolist()

    # Encodes the new categorical variables
    datos = pd.get_dummies(datos[variables_total], columns=var_categ_new, drop_first=True)

    # Calculates the interaction in the case that they are specified
    if len(var_interac) > 0:
        n_var = len(datos.columns)
        for interac in var_interac:
            vars1 = [x for x in datos.columns[:n_var] if interac[0] in x]
            vars2 = [x for x in datos.columns[:n_var] if interac[1] in x]
            combinaciones = list(product(vars1, vars2))
            for comb in combinaciones:
                datos[comb[0] + '_' + comb[1]] = datos[comb[0]] * datos[comb[1]]

        # Select the interaction variables
        datos_interac = datos.iloc[:, list(range(n_var, len(datos.columns)))]

        # Concatenate variables
        return pd.concat([datos_v, datos_interac], axis=1)

    # If there are no interactions returns the same data
    return datos_v


def Rsq(modelo, varObj, datos):
    """
    Calculates the coeficient of determination (R-squared) of a model of linea regression.

    Parameters:
    modelo (RegressionResultsWrapper): Linear regression model adjusted.
    varObj (Series o array): Target numerical variable to predict.
    datos (DataFrame): Dataframe for which we want to evaluate the model.

    Returns:
    float: Coeficient of determination (R-squared) of the model.
    """

    # Selects the independent variables of the model
    datos = datos[modelo.model.exog_names[1:]]

    # Predictions using the model
    testpredicted = modelo.predict(sm.add_constant(datos))

    # Calculates the sum of the errors (SEE)
    sse = sum((testpredicted - varObj) ** 2)

    # Calculates the total sum of the squares (SST)
    sst = sum((varObj - varObj.mean()) ** 2)

    # Calculates and returns the coeficient of determination (R-squared)
    return 1 - sse / sst


# Para evaluar el pseudo-R2 en regr. logistica en cualquier conjunto de datos
def pseudoR2(modelo, dd, target):
    """
    Calculates the pseudo R2 for a logistic regression model.

    Parámetros:
    modelo (LogisticRegression): Logistic regression model adjusted.
    dd (DataFrame): Dataframe for which we want to do the predictions.
    target (Series o array): Target variable we want to predict.

    Returns:
    float: Value of pseudo R squared for the logistic regression model.
    """

    # Does the predictions of probability of success of the model with the data input.
    pred_out_link = modelo.predict_proba(dd)[:, 1]

    # Adjusts a null model and calculates the null predictions
    mod_out_null = sm.Logit(target, np.ones(len(dd))).fit()
    pred_out_linkN = mod_out_null.predict(np.ones(len(dd)))

    # Calculates the numerator of pseudo R-squared
    num = np.sum((target == 1) * np.log(pred_out_link) + np.log(1 - pred_out_link) * (1 - (target == 1)))

    # Calculates the denominator
    denom = np.sum((target == 1) * np.log(pred_out_linkN) + np.log(1 - pred_out_linkN) * (1 - (target == 1)))

    # Calculation of pseudo R2
    pseudo_R2 = 1 - (num / denom)
    return pseudo_R2


def lm(varObjCont, datos, var_cont, var_categ, var_interac=[]):
    """
    Adjusts a model of linear regression to the data and returns information related to the model.

    Parameters:
    varObjCont (Series o array): Contunious target variable, trying to predict.
    datos (DataFrame): Dataframe that contains the input variable.
    var_cont (lista): List with the names of the continous variable.
    var_categ (lista): List with the names of the categorical variables.
    var_interac (lista, opcional): List of paur of variables for the interaction (by defect a null list)

    Returns:
    dict: A dictionary that contains information related with the adjusted model, inludint the model itself,
    the list of continuous and categorica variables, the interaction variables (if specified) and the Dataframe X
    used to make a model.
    """
    # Prepares the data for the model.
    datos = crear_data_modelo(datos, var_cont, var_categ, var_interac)

    # Adjusts a model of linear regression to the data and stores the information of the model in 'Modelo'
    output = {
        'Modelo': sm.OLS(varObjCont, sm.add_constant(datos)).fit(),
        'Variables': {
            'cont': var_cont,
            'categ': var_categ,
            'inter': var_interac
        },
        'X': datos
    }

    return output


# Funcion para hacer validacion cruzada a variable respuesta continua
def validacion_cruzada_lm(n_cv, datos, varObjCont, var_cont, var_categ, var_interac=[]):
    """
    Performs k-fold cross-validation for a linear regression model and returns a list of R-squared scores for each fold

    Parámetros:
    n_cv (int): The number of divisions for the cross validation
    datos (DataFrame): El DataFrame of input data variables.
    varObjCont (Series o array): Target variable.
    var_cont (lista): List of continous variables names.
    var_categ (lista): List of categorical variables names.
    var_interac (lista, opcional): List of pairs of variables with interactions, by defect is null.

    Returns:
    list: List of R-squared scores that are obtained through the cross validation
    """
    
    # Prepares the data for the model
    datos = crear_data_modelo(datos, var_cont, var_categ, var_interac)

    # Does cross validation
    return list(cross_val_score(LinearRegression(), datos, varObjCont, cv=n_cv, scoring='r2'))



def modelEffectSizes(modelo, varObjCont, datos, var_cont, var_categ, var_interac=[]):
    """
    Calculats R-squared from a model and presents them graphically.

    Parameters:
    modelo (dict): Dictionary that contains the model and other informatiomn.
    varObjCont (Series o array): Target continuous variable.
    datos (DataFrame): Dtaframe with data input.
    var_cont (lista): List of cotinuous variable names.
    var_categ (lista): List of names with categorical names.
    var_interac (lista, opcional): List of variables for which we want to check the interaction, by defect is null.

    Returns:
    DataFrame: Dataframe with the aportation of each variable to the R-squared.
    """

    # Created a complete model and calculates de R-squared
    modelo_completo = lm(varObjCont, datos, var_cont, var_categ, var_interac)
    r2_completo = Rsq(modelo_completo['Modelo'], varObjCont, modelo_completo['X'])
    
    # Initializes lists to store variables names and their aportations to the R-squared.
    variables = []
    r2 = []

    # Calculates the contibution of the R-squared of the continuous variables.
    for x in var_cont:
        variables.append(x)
        var = [v for v in var_cont if v != x]
        modelo2 = lm(varObjCont, datos, var, var_categ, var_interac)
        r2.append(r2_completo - Rsq(modelo2['Modelo'], varObjCont, modelo2['X']))

    #Aporation from the R-squared to the categorical variables
    for x in var_categ:
        variables.append(x)
        var = [v for v in var_categ if v != x]
        modelo2 = lm(varObjCont, datos, var_cont, var, var_interac)
        r2.append(r2_completo - Rsq(modelo2['Modelo'], varObjCont, modelo2['X']))    

    # Calculates R-squared coming from interaction variables.
    for x in var_interac:
        variables.append(x[0] + '_' + x[1])
        var = [v for v in var_interac if v != x]
        modelo2 = lm(varObjCont, datos, var_cont, var_categ, var)
        r2.append(r2_completo - Rsq(modelo2['Modelo'], varObjCont, modelo2['X']))    

    # Creates a Dataframe with each aportation
    aportacion_r2 = pd.DataFrame({
        'Variables': variables,
        'R2': r2
    })
    aportacion_r2 = aportacion_r2.sort_values('R2')

    # Bar graph with contribution for each variable.
    plt.figure(figsize=(10, 6))
    plt.barh(aportacion_r2['Variables'], aportacion_r2['R2'], color='skyblue')
    plt.xlabel('Aportacion R2')
    plt.show()

    return aportacion_r2


def impVariablesLog(modelo, varObjBin, datos, var_cont, var_categ, var_interac = []):
    """
    Same as above, but for the lofistic regression model.

    Parameters:
        modelo (dict): Un diccionario que contiene el modelo ajustado, las variables utilizadas y la matriz de características.
        varObjBin (array-like): Variable objetivo binaria.
        datos (DataFrame): Conjunto de datos que incluye las variables predictoras.
        var_cont (list): Lista de nombres de variables continuas.
        var_categ (list): Lista de nombres de variables categóricas.
        var_interac (list, opcional): Lista de interacciones entre variables.

    Returns:
        DataFrame: Un DataFrame que muestra la importancia de cada variable en función de su contribución al R2.
    """
    # Ajusta un modelo de regresión logística completo y calcula el R2 completo.
    modelo_completo = glm(varObjBin, datos, var_cont, var_categ, var_interac)
    r2_completo = pseudoR2(modelo_completo['Modelo'], modelo_completo['X'], varObjBin)
    
    # Inicializa listas para almacenar variables y sus respectivas contribuciones al R2.
    variables = []
    r2 = []
    
    # Evalúa el impacto de eliminar variables continuas en el modelo.
    for x in var_cont:
        variables.append(x)
        var = [v for v in var_cont if v != x]
        modelo2 = glm(varObjBin, datos, var, var_categ, var_interac)
        r2.append(r2_completo - pseudoR2(modelo2['Modelo'], modelo2['X'], varObjBin))
    
    # Evalúa el impacto de eliminar variables categóricas en el modelo.
    for x in var_categ:
        variables.append(x)
        var = [v for v in var_categ if v != x]
        modelo2 = glm(varObjBin, datos, var_cont, var, var_interac)
        r2.append(r2_completo - pseudoR2(modelo2['Modelo'], modelo2['X'], varObjBin))
    
    # Evalúa el impacto de eliminar variables de interacción en el modelo.
    for x in var_interac:
        variables.append(x[0] + '_' + x[1])
        var = [v for v in var_interac if v != x]
        modelo2 = glm(varObjBin, datos, var_cont, var_categ, var)
        r2.append(r2_completo - pseudoR2(modelo2['Modelo'], modelo2['X'], varObjBin))
    
    # Crea un DataFrame con la contribución R2 de cada variable.
    aportacion_r2 = pd.DataFrame({
        'Variables': variables,
        'R2': r2
    })
    
    # Ordena el DataFrame por la contribución R2.
    aportacion_r2 = aportacion_r2.sort_values('R2')
    
    # Imprime el DataFrame y genera un gráfico de barras.
    print(aportacion_r2)
    plt.figure(figsize=(10, 6))
    plt.barh(aportacion_r2['Variables'], aportacion_r2['R2'], color='skyblue')
    plt.xlabel('Aportación R2')
    plt.show()
    
    return aportacion_r2


def glm(varObjBin, datos, var_cont, var_categ, var_interac = []):
    """
    Adjusts a logistic linear regression model to the data.

    Parameters:
        varObjBin (array-like): Variable objetivo binaria.
        datos (DataFrame): Conjunto de datos que incluye las variables predictoras.
        var_cont (list): Lista de nombres de variables continuas.
        var_categ (list): Lista de nombres de variables categóricas.
        var_interac (list, opcional): Lista de interacciones entre variables (por defecto es una lista vacía)..

    Returns:
        dict: Un diccionario que contiene el modelo ajustado, las variables utilizadas y el conjunto de datos utilizado en la predicción.
    """

    # Preprocesar los datos aplicando la función crear_data_modelo
    datos = crear_data_modelo(datos, var_cont, var_categ, var_interac)
    
    # Crear un modelo de regresión logística y ajustarlo a los datos
    output = {
        'Modelo': LogisticRegression(max_iter=1000, solver='newton-cg').fit(datos, varObjBin),
        'Variables': {
            'cont': var_cont,
            'categ': var_categ,
            'inter': var_interac
        },
        'X': datos
    }
    
    return output



def summary_glm(modelo, varObjBin, datos):
    """
    Results for the logistic regression model.

    Parameters:
        modelo (LogisticRegression): Modelo de regresión logística previamente ajustado.
        varObjBin (array-like): Variable objetivo binaria.
        datos (DataFrame): Conjunto de datos que incluye las variables predictoras.

    Returns:
        dict: Un diccionario que contiene el resumen de los resultados del modelo, incluyendo coeficientes,
        estadísticas y medidas de ajuste.
    """
    # Realiza predicciones de probabilidad utilizando el modelo.
    p = modelo.predict_proba(datos)

    # Identifica y elimina las columnas con un solo valor único, ya que no aportan información.
    eliminar = [x for x in range(len(datos.columns)) if len(set(datos[datos.columns[x]])) == 1]
    datos = datos.drop(datos.columns[eliminar], axis=1)

    # Prepara el diseño de la matriz de características.
    # Crea una matriz de diseño que incluye un término de intercepción (columna de unos).
    X_design = np.hstack([np.ones((datos.shape[0], 1)), datos])

    # Calcula la matriz de covarianza.
    V = np.diagflat(np.product(p, axis=1))
    matriz_cov = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))

    # Obtiene los coeficientes del modelo ajustado, excluyendo las columnas eliminadas.
    coefs = [modelo.coef_[0][x] for x in range(len(modelo.coef_[0])) if x not in eliminar]

    # Crea un DataFrame con el resumen de los coeficientes y estadísticas.
    output = pd.DataFrame({
        'Variable': ['(Intercept)'] + list(datos.columns),
        'Estimate': [modelo.intercept_[0]] + coefs,
        'Std. Error': np.sqrt(np.diag(matriz_cov))
    })

    # Calcula estadísticas adicionales como el valor z, p-valor y significancia.
    output['z value'] = output['Estimate'] / output['Std. Error'] #*
    output['p value'] = 2 * (1 - stats.t.cdf(abs(output['z value']), df=len(datos) - len(modelo.coef_[0])))
    output['signif'] = [''] * len(output)
    output.loc[output['p value'] < 0.1, 'signif'] = '.'
    output.loc[output['p value'] < 0.05, 'signif'] = '*'
    output.loc[output['p value'] < 0.01, 'signif'] = '**'
    output.loc[output['p value'] < 0.001, 'signif'] = '***'
    output['p value'] = list(map(lambda x: "{:f}".format(x), output['p value']))

    # Calcula la verosimilitud, el AIC y el BIC del modelo.
    y = list(varObjBin)
    y_pred = p[:, 1]
    sum1 = sum([int(y[i]) * np.log(y_pred[i] / (1 - y_pred[i])) for i in range(len(datos))])
    sum2 = sum([np.log(1 - y_pred[i]) for i in range(len(datos))])
    verosimilitud = sum1 + sum2
    aic = 2 * (len(modelo.coef_) + 1) - 2 * verosimilitud
    bic = np.log(len(datos)) * (len(modelo.coef_) + 1) - 2 * verosimilitud

    # Retorna un diccionario que contiene el resumen de resultados.
    output = {
        'Contrastes': output,
        'BondadAjuste': pd.DataFrame({'LLK': [verosimilitud], 'AIC': [aic], 'BIC': [bic]})
    }
	
    return output

def validacion_cruzada_glm(n_cv, datos, varObjBin, var_cont, var_categ, var_interac = []):
    """
    Cross-validation for glm.

    Parameters:
        n_cv (int): Número de particiones en la validación cruzada (k-fold).
        datos (DataFrame): Conjunto de datos que incluye las variables predictoras.
        varObjBin (array-like): Variable objetivo binaria.
        var_cont (list): Lista de nombres de variables continuas.
        var_categ (list): Lista de nombres de variables categóricas.
        var_interac (list, opcional): Lista de interacciones entre variables.

    Returns:
        list: Una lista de puntuaciones ROC AUC obtenidas en cada partición de la validación cruzada.
    """
    # Prepara los datos según las variables de entrada y las interacciones.
    datos = crear_data_modelo(datos, var_cont, var_categ, var_interac)
	
    
    # Realiza la validación cruzada utilizando el modelo de regresión logística y calcula el ROC AUC.
    return list(cross_val_score(LogisticRegression(max_iter=1000, solver='newton-cg'), datos, varObjBin, cv = n_cv, scoring = 'roc_auc'))



def sensEspCorte(modelo, dd, varObjBin, ptoCorte, var_cont, var_categ, var_interac = []):
    """
    Calculates quality measures for a given point.

    Parameters:
        modelo (dict): Diccionario que contiene el modelo ajustado.
        dd (DataFrame): Conjunto de datos de prueba.
        varObjBin (array-like): Variable objetivo binaria.
        ptoCorte (float): Punto de corte para la clasificación.
        var_cont (list): Lista de nombres de variables continuas.
        var_categ (list): Lista de nombres de variables categóricas.
        var_interac (list, opcional): Lista de interacciones entre variables.

    Returns:
        DataFrame: Un DataFrame que contiene las medidas de calidad para el punto de corte dado.
    """
    # Prepara los datos de prueba según el modelo
    if len(var_interac) > 0:
        dd = crear_data_modelo(dd, var_cont, var_categ, var_interac)
    else:
        dd = pd.get_dummies(dd[var_cont + var_categ], columns=var_categ, drop_first=True)
    
    # Calcula las probabilidades de la clase positiva
    probs = modelo.predict_proba(dd)[:, 1]
    
    # Realiza la clasificación en función del punto de corte
    preds = (probs > ptoCorte).astype(int)
    
    # Calcula la matriz de confusión
    cm = confusion_matrix(varObjBin, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcula medidas de calidad
    output = pd.DataFrame({
        'PtoCorte': [ptoCorte],
        'Accuracy': [(tp + tn) / (tn + fp + fn + tp)],
        'Sensitivity': [tp / (tp + fn)],
        'Specificity': [tn / (tn + fp)],
        'PosPredValue': [tp / (tp + fp)],
        'NegPredValue': [tn / (tn + fn)]
    })
    
    return output



def lm_forward(varObjCont, datos, var_cont, var_categ, var_interac = [], metodo = 'AIC'):
    """
    Selection of varibales *onwards* for a linear regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.
    
    Argumentos de entrada:
    - varObjCont: La variable objetivo (dependiente) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').
    
    Argumentos de salida:
    - Un modelo de regresión lineal que utiliza el mejor conjunto de variables encontrado.
    """
    
    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac
    # Inicializar listas para almacenar las variables seleccionadas.
    var_cont_final, var_categ_final, var_interac_final = [], [], []

    # Definir la función 'calcular_metrica' dependiendo del método de bondad de ajuste selecionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo):
            return 2 * (modelo.df_model + 1) - 2 * modelo.llf
    elif metodo == 'BIC':
        def calcular_metrica(modelo):
            return np.log(len(datos)) * (modelo.df_model + 1) - 2 * modelo.llf

    # Ajustar un modelo inicial con una sola constante.
    modelo_inicial = sm.OLS(varObjCont, np.ravel([1] * len(datos))).fit()
    metrica_inicial = calcular_metrica(modelo_inicial)
    dif_metrica = 1

    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ 1')
    print('')
    
    # Comenzar el bucle de selección de variables.
    while((dif_metrica > 0) and (len(variables) > 0)):
        variables_probar = []
        metricas = []
        
        # Iterar a través de las variables restantes.
        for x in variables:
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = var_cont_final + [x]
            elif x in var_categ:
                var_categ_probar = var_categ_final + [x]
            else:
                var_interac_probar = var_interac_final + [x]
            
            # Ajustar un modelo con la variable actual y calcular la métrica.
            modelo = lm(varObjCont, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo))
        
        # Imprimir métricas de AIC/BIC para las variables probadas.
        print(pd.DataFrame({
            'Variable': [' + ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo).to_string(index = False))
        print('')
        
        # Elegir la mejor variable y su métrica actual.
        mejor_variable = variables_probar[min(enumerate(metricas), key = lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key = lambda x: x[1])[0]]

        # Agregar la mejor variable al conjunto final apropiado.
        if mejor_variable in var_cont:
            var_cont_final.append(mejor_variable)
        elif mejor_variable in var_categ:
            var_categ_final.append(mejor_variable)
        else:
            var_interac_final.append(mejor_variable)
        
        # Actualizar la diferencia en métrica, las variables y el modelo inicial.
        dif_metrica = metrica_inicial - metrica_actual
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual
        modelo_inicial = modelo
        
        # Gestionar la eliminación de variables si la métrica no mejora.
        if dif_metrica <= 0:
            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
        else:
            # Imprimir el modelo y la métrica actual si la métrica mejora.
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Forward: Entra ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')
    
    # Devolver el modelo final con el conjunto de variables seleccionado.
    return lm(varObjCont, datos, var_cont_final, var_categ_final, var_interac_final)



def lm_backward(varObjCont, datos, var_cont, var_categ, var_interac = [], metodo = 'AIC'):
    """

    Selection of varibales *backwards* for a linear regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.
    
    Argumentos de entrada:
    - varObjCont: La variable objetivo (dependiente) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').
    
    Argumentos de salida:
    - Un modelo de regresión lineal que utiliza el mejor conjunto de variables encontrado.
    """
    
    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac
    # Crea listas con todas las variables con las que empezar el proceso
    var_cont_final, var_categ_final, var_interac_final = var_cont, var_categ, var_interac

    # Definir la función 'calcular_metrica' dependiendo del método seleccionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo):
            return 2 * (modelo.df_model + 1) - 2 * modelo.llf
    elif metodo == 'BIC':
        def calcular_metrica(modelo):
            return np.log(len(datos)) * (modelo.df_model + 1) - 2 * modelo.llf

    # Ajustar un modelo inicial con todas las variables.
    modelo_inicial = lm(varObjCont, datos, var_cont, var_categ, var_interac)['Modelo']
    metrica_inicial = calcular_metrica(modelo_inicial)
    dif_metrica = -0.1
    
    # Crear la fórmula inicial con todas las variables.
    formula = ' + '.join(var_cont + var_categ + ['*'.join(x) for x in var_interac])
    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ ' + formula)
    print('')
    
    # Comenzar el bucle de selección de variables hacia atrás.
    while((dif_metrica <= 0) and (len(variables) > 0)):
		
        variables_probar = []
        metricas = []
        
        # Iterar a través de las variables restantes.
        for x in variables:
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = [v for v in var_cont_probar if v != x]
            elif x in var_categ:
                var_categ_probar = [v for v in var_categ_probar if v != x]
            else:
                var_interac_probar = [v for v in var_interac_probar if v != x]
            
            # Ajustar un modelo sin la variable actual y calcular la métrica.
            modelo = lm(varObjCont, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo))
        
        # Imprimir métricas de AIC/BIC para las variables probadas.
        print(pd.DataFrame({
            'Variable': [' - ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo, ascending = False).to_string(index = False))
        print('')
        
        # Elegir la mejor variable a eliminar y su métrica actual.
        mejor_variable = variables_probar[min(enumerate(metricas), key = lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key = lambda x: x[1])[0]]
        
        # Eliminar la mejor variable del conjunto final apropiado.
        if mejor_variable in var_cont:
            var_cont_final = [x for x in var_cont_final if x != mejor_variable]
        elif mejor_variable in var_categ:
            var_categ_final = [x for x in var_categ_final if x != mejor_variable]
        else:
            var_interac_final = [x for x in var_interac_final if x != mejor_variable]
        
        # Calcular la diferencia en métrica y actualizar las variables.
        dif_metrica = metrica_actual - metrica_inicial
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual
        
        # Gestionar la adición de variables si la métrica mejora.
        if dif_metrica > 0:
            if mejor_variable in var_cont:
                var_cont_final.append(mejor_variable)
            elif mejor_variable in var_categ:
                var_categ_final.append(mejor_variable)
            else:
                var_interac_final.append(mejor_variable)
        else:
            # Imprimir el modelo y la métrica actual si la métrica no mejora.
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Backward: Sale ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')
    
    # Devolver el modelo final con el conjunto de variables seleccionado.
    return lm(varObjCont, datos, var_cont_final, var_categ_final, var_interac_final)




def lm_stepwise(varObjCont, datos, var_cont, var_categ, var_interac = [], metodo='AIC'):
    """
   Selection of varibales *stepwise* for a linear regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.

    Argumentos de entrada:
    - varObjCont: La variable objetivo (dependiente) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').

    Argumentos de salida:
    - Un modelo de regresión lineal que utiliza el mejor conjunto de variables encontrado.
    """

    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac

    # Inicializar listas para almacenar las variables seleccionadas.
    var_cont_final, var_categ_final, var_interac_final = [], [], []

    # Definir la función 'calcular_metrica' dependiendo del método seleccionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo):
            return 2 * (modelo.df_model + 1) - 2 * modelo.llf
    elif metodo == 'BIC':
        def calcular_metrica(modelo):
            return np.log(len(datos)) * (modelo.df_model + 1) - 2 * modelo.llf

    # Ajustar un modelo inicial con una sola constante.
    modelo_inicial = sm.OLS(varObjCont, np.ravel([1] * len(datos))).fit()
    metrica_inicial = calcular_metrica(modelo_inicial)
    dif_metrica = 1

    # Imprimir información inicial.
    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ 1')
    print('')

    while((dif_metrica > 0) and (len(variables) > 0)):

        # Entrada variable (step forward)
        variables_probar = []
        metricas = []
        for x in variables:
            # Crear copias de las listas de variables para probar cambios sin modificar las originales.
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = var_cont_final + [x]
            elif x in var_categ:
                var_categ_probar = var_categ_final + [x]
            else:
                var_interac_probar = var_interac_final + [x]
            # Ajustar un modelo sin la variable actual y calcular la métrica.
            modelo = lm(varObjCont, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo))
        # Imprimir las métricas para las variables probadas.
        print(pd.DataFrame({
            'Variable': [' + ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo).to_string(index=False))
        print('')
        # Elegir la mejor variable a agregar y su métrica actual.
        mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

        if mejor_variable in var_cont:
            var_cont_final.append(mejor_variable)
        elif mejor_variable in var_categ:
            var_categ_final.append(mejor_variable)
        else:
            var_interac_final.append(mejor_variable)
        # Calcular la diferencia en métrica y actualizar las variables.
        dif_metrica = metrica_inicial - metrica_actual
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual
        modelo_inicial = modelo

        if dif_metrica <= 0:
            # Si la métrica no mejora, eliminar la variable recién agregada.
            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
        else:
            # Si la métrica mejora, imprimir el paso y la métrica actual.
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Forward: Entra ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')

        # Salida de variable (step backward)
        if ((dif_metrica > 0) and (len(var_cont_final + var_categ_final + var_interac_final) > 1)):
            variables_probar = []
            metricas = []
            for x in var_cont_final + var_categ_final + var_interac_final:
                # Crear copias de las listas de variables para probar cambios sin modificar las originales.
                var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
                if x in var_cont:
                    var_cont_probar = [v for v in var_cont_probar if v != x]
                elif x in var_categ:
                    var_categ_probar = [v for v in var_categ_probar if v != x]
                else:
                    var_interac_probar = [v for v in var_interac_probar if v != x]
                # Ajustar un modelo sin la variable actual y calcular la métrica.
                modelo = lm(varObjCont, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
                variables_probar.append(x)
                metricas.append(calcular_metrica(modelo))
            # Imprimir las métricas para las variables probadas.
            print(pd.DataFrame({
                'Variable': [' - ' + str(x) for x in variables_probar], metodo: metricas
                }).sort_values(metodo, ascending=False).to_string(index=False))
            print('')
            # Elegir la mejor variable a eliminar y su métrica actual.
            mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
            metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
            # Calcular la diferencia en métrica y actualizar las variables.
            dif_metrica_salida = metrica_actual - metrica_inicial
            variables = [x for x in variables if x != mejor_variable]

            if dif_metrica_salida < 0:
                # Si la métrica no mejora, agregar la variable recién eliminada.
                if mejor_variable in var_cont:
                    var_cont_final = [x for x in var_cont_final if x != mejor_variable]
                elif mejor_variable in var_categ:
                    var_categ_final = [x for x in var_categ_final if x != mejor_variable]
                else:
                    var_interac_final = [x for x in var_interac_final if x != mejor_variable]
                modelo_inicial = modelo
                metrica_inicial = metrica_actual
                formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
                print('--------------------------------------- Step Backward: Sale ' + str(mejor_variable))
                print('')
                print('AIC = ' + str(metrica_inicial))
                print('')
                print('y ~ ' + formula)
                print('')
            else:
                # Si la métrica mejora, imprimir el paso y la métrica actual.
                if mejor_variable in var_cont:
                    var_cont_final.append(mejor_variable)
                elif mejor_variable in var_categ:
                    var_categ_final.append(mejor_variable)
                else:
                    var_interac_final.append(mejor_variable)
                formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
                print('--------------------------------------- Step Backward: No sale ninguna')
                print('')
                print('AIC = ' + str(metrica_inicial))
                print('')
                print('y ~ ' + formula)
                print('')

    # Devolver el modelo final con el conjunto de variables seleccionado.
    return lm(varObjCont, datos, var_cont_final, var_categ_final, var_interac_final)



																   
																   
def glm_forward(varObjBin, datos, var_cont, var_categ, var_interac = [], metodo = 'AIC'):
    """
    Selection of varibales *onwards* for a *logistic* regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.

    Argumentos de entrada:
    - varObjBin: La variable objetivo binaria (0 o 1) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').

    Argumentos de salida:
    - Un modelo de regresión logística que utiliza el mejor conjunto de variables encontrado.
    """

    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac

    # Inicializar listas para almacenar las variables seleccionadas.
    var_cont_final, var_categ_final, var_interac_final = [], [], []

    # Función para calcular la log-verosimilitud de un modelo con ciertas variables.
    def calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac):
        y=list(y)
        x = crear_data_modelo(x, var_cont, var_categ, var_interac)
        if len(x) == 0:
            x = np.array([1]*len(datos)).reshape(-1, 1)
        y_pred = modelo.predict_proba(x)[:, 1]
        sum1 = sum([int(y[i])*np.log(y_pred[i]/(1 - y_pred[i])) for i in range(len(x))])
        sum2 = sum([np.log(1 - y_pred[i]) for i in range(len(x))])
        return sum1 + sum2

    # Definir la función 'calcular_metrica' dependiendo del método seleccionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return 2*(len(modelo.coef_) + 1) - 2*calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac)
    elif metodo == 'BIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return np.log(len(datos))*(len(modelo.coef_) + 1) - 2*calcular_logverosimilitud(modelo, x, y, 
                                                                                            var_cont, var_categ, 
                                                                                            var_interac)

    # Crear un conjunto de entrenamiento con una constante.
    train = np.array([1]*len(datos)).reshape(-1, 1)
    
    # Ajustar un modelo inicial sin ninguna variable.
    modelo_inicial = LogisticRegression(fit_intercept=False, max_iter=1000, solver='newton-cg').fit(train, varObjBin)
    metrica_inicial = calcular_metrica(modelo_inicial, train, varObjBin, [], [], [])
    dif_metrica = 3.1

    # Imprimir información inicial.
    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ 1')
    print('')
    while((dif_metrica > 3) and (len(variables) > 0)): #**a mayor número mas exigente para entrar

        # Entrada variable (step forward)
        variables_probar = []
        metricas = []
        for x in variables:
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = var_cont_final + [x]
            elif x in var_categ:
                var_categ_probar = var_categ_final + [x]
            else:
                var_interac_probar = var_interac_final + [x]
            # Ajustar un modelo con la variable actual y calcular la métrica.
            modelo = glm(varObjBin, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo, datos, varObjBin, var_cont_probar, 
                                             var_categ_probar, var_interac_probar))
        # Imprimir las métricas para las variables probadas.
        print(pd.DataFrame({
            'Variable': [' + ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo).to_string(index=False))
        print('')
        # Elegir la mejor variable a agregar y su métrica actual.
        mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

        # Agregar la mejor variable al conjunto final.
        if mejor_variable in var_cont:
            var_cont_final.append(mejor_variable)
        elif mejor_variable in var_categ:
            var_categ_final.append(mejor_variable)
        else:
            var_interac_final.append(mejor_variable)
        dif_metrica = metrica_inicial - metrica_actual
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual
        modelo_inicial = modelo
        if dif_metrica <= 3: #**a mayor número mas exigente para entrar
            # Si la métrica no mejora, eliminar la variable recién agregada.
            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
        else:
            # Si la métrica mejora, imprimir el paso y la métrica actual.
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Forward: Entra ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')

    # Devolver el modelo final con el conjunto de variables seleccionado.
    return glm(varObjBin, datos, var_cont_final, var_categ_final, var_interac_final)




def glm_backward(varObjBin, datos, var_cont, var_categ, var_interac = [], metodo = 'AIC'):
    """
    Selection of varibales *backwards* for a logistic regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.

    Argumentos de entrada:
    - varObjBin: La variable objetivo binaria (0 o 1) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').

    Argumentos de salida:
    - Un modelo de regresión logística que utiliza el mejor conjunto de variables encontrado.
    """

    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac

    # Inicializar listas para almacenar las variables seleccionadas.
    var_cont_final, var_categ_final, var_interac_final = var_cont, var_categ, var_interac

    # Función para calcular la log-verosimilitud de un modelo con ciertas variables.
    def calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac):
        y=list(y)
        x = crear_data_modelo(x, var_cont, var_categ, var_interac)
        if len(x) == 0:
            x = np.array([1]*len(datos)).reshape(-1, 1)
        y_pred = modelo.predict_proba(x)[:, 1]
        sum1 = sum([int(y[i])*np.log(y_pred[i]/(1 - y_pred[i])) for i in range(len(x))])
        sum2 = sum([np.log(1 - y_pred[i]) for i in range(len(x))])
        return sum1 + sum2

    # Definir la función 'calcular_metrica' dependiendo del método seleccionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return 2*(len(modelo.coef_) + 1) - 2*calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac)
    elif metodo == 'BIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return np.log(len(datos))*(len(modelo.coef_) + 1) - 2*calcular_logverosimilitud(modelo, x, y, 
                                                                                            var_cont, var_categ, 
                                                                                            var_interac)

    # Ajustar un modelo inicial con todas las variables.
    modelo_inicial = glm(varObjBin, datos, var_cont, var_categ, var_interac)['Modelo']
    metrica_inicial = calcular_metrica(modelo_inicial, datos, varObjBin, var_cont, var_categ, var_interac)
    dif_metrica = -0.1

    # Crear una fórmula inicial.
    formula = ' + '.join(var_cont + var_categ + ['*'.join(x) for x in var_interac])

    # Imprimir información inicial.
    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ ' + formula)
    print('')

    while((dif_metrica <= 10) and (len(variables) > 0)): # **aumentando el numero soy mas exigente para agregar variables

        # Variables a probar (step backward).
        variables_probar = []
        metricas = []
        for x in variables:
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = [v for v in var_cont_probar if v != x]
            elif x in var_categ:
                var_categ_probar = [v for v in var_categ_probar if v != x]
            else:
                var_interac_probar = [v for v in var_interac_probar if v != x]
            # Ajustar un modelo sin la variable actual y calcular la métrica.
            modelo = glm(varObjBin, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo, datos, varObjBin, var_cont_probar, var_categ_probar, 
                                             var_interac_probar))
            

        # Imprimir las métricas para las variables probadas.
        print(pd.DataFrame({
            'Variable': [' - ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo, ascending=False).to_string(index=False))
        print('')
        # Elegir la mejor variable a eliminar y su métrica actual.
        mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

        # Eliminar la mejor variable del conjunto final.
        if mejor_variable in var_cont:
            var_cont_final = [x for x in var_cont_final if x != mejor_variable]
        elif mejor_variable in var_categ:
            var_categ_final = [x for x in var_categ_final if x != mejor_variable]
        else:
            var_interac_final = [x for x in var_interac_final if x != mejor_variable]

        dif_metrica = metrica_actual - metrica_inicial
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual

        if dif_metrica > 10: # aumentando el numero soy mas exigente para agregar variables igual que **
            # Si la métrica mejora, agregar la variable eliminada nuevamente.
            if mejor_variable in var_cont:
                var_cont_final.append(mejor_variable)
            elif mejor_variable in var_categ:
                var_categ_final.append(mejor_variable)
            else:
                var_interac_final.append(mejor_variable)
        else:
            # Si la métrica no mejora, imprimir el paso y la métrica actual.
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Backward: Sale ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')
			
																					

    # Devolver el modelo final con el conjunto de variables seleccionado.
    return glm(varObjBin, datos, var_cont_final, var_categ_final, var_interac_final)


def glm_stepwise(varObjBin, datos, var_cont, var_categ, var_interac = [], metodo='AIC'):
    """
    Selection of varibales *stepwise* for a logistic regression model. The goal is to find the best set of variables to predict the target variable
    using AIC or BIC criteria.

    Argumentos de entrada:
    - varObjBin: La variable objetivo binaria (0 o 1) que deseamos predecir.
    - datos: El DataFrame que contiene todas las variables.
    - var_cont: Una lista de nombres de variables continuas.
    - var_categ: Una lista de nombres de variables categóricas.
    - var_interac: Una lista de nombres de variables de interacción (por defecto, una lista vacía).
    - metodo: El criterio para seleccionar variables ('AIC' o 'BIC', por defecto 'AIC').

    Argumentos de salida:
    - Un modelo de regresión logística que utiliza el mejor conjunto de variables encontrado.
    """

    # Crear una lista 'variables' que contiene todas las variables a considerar.
    variables = var_cont + var_categ + var_interac

    # Inicializar listas para almacenar las variables seleccionadas.
    var_cont_final, var_categ_final, var_interac_final = [], [], []

    # Función para calcular la log-verosimilitud de un modelo con ciertas variables.
    def calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac):
        y=list(y)
        x = crear_data_modelo(x, var_cont, var_categ, var_interac)
        if len(x) == 0:
            x = np.array([1]*len(datos)).reshape(-1, 1)
        y_pred = modelo.predict_proba(x)[:, 1]
        sum1 = sum([int(y[i])*np.log(y_pred[i]/(1 - y_pred[i])) for i in range(len(x))])
        sum2 = sum([np.log(1 - y_pred[i]) for i in range(len(x))])
        #print('logVerosimilutud = ', sum1 +sum2)
        return sum1 + sum2

    # Definir la función 'calcular_metrica' dependiendo del método seleccionado.
    if metodo == 'AIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return 2 * (len(modelo.coef_) + 1) - 2 * calcular_logverosimilitud(modelo, x, y, var_cont, var_categ, var_interac)
    elif metodo == 'BIC':
        def calcular_metrica(modelo, x, y, var_cont, var_categ, var_interac):
            return np.log(len(datos)) * (len(modelo.coef_) + 1) - 2 * calcular_logverosimilitud(modelo, x, y,
                                                                                                var_cont, var_categ,
                                                                                                var_interac)

    # Ajustar un modelo inicial con una constante (intercept).
    train = np.array([1] * len(datos)).reshape(-1, 1)
    modelo_inicial = LogisticRegression(fit_intercept=False, max_iter=1000, solver='newton-cg').fit(train, varObjBin)
    metrica_inicial = calcular_metrica(modelo_inicial, train, varObjBin, [], [], [])
    dif_metrica = 5.1

    # Imprimir información inicial.
    print('Start: ' + metodo + ' = ' + str(metrica_inicial))
    print('')
    print('y ~ 1')
    print('')

    while((dif_metrica > 5) and (len(variables) > 0)): #Aumentando el numero soy mas exigente para la entrada (igual que **)

        # Etapa de avance (entrada de variable)
        variables_probar = []
        metricas = []
        for x in variables:
            var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
            if x in var_cont:
                var_cont_probar = var_cont_final + [x]
            elif x in var_categ:
                var_categ_probar = var_categ_final + [x]
            else:
                var_interac_probar = var_interac_final + [x]
            # Ajustar un modelo con la variable actual y calcular la métrica.
            modelo = glm(varObjBin, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
            variables_probar.append(x)
            metricas.append(calcular_metrica(modelo, datos, varObjBin, var_cont_probar,
                                             var_categ_probar, var_interac_probar))
        # Imprimir las métricas para las variables probadas en la etapa de avance.
        print(pd.DataFrame({
            'Variable': [' + ' + str(x) for x in variables_probar], metodo: metricas
            }).sort_values(metodo).to_string(index=False))
        print('')
        mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
        metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

        # Actualizar las variables seleccionadas.
        if mejor_variable in var_cont:
            var_cont_final.append(mejor_variable)
        elif mejor_variable in var_categ:
            var_categ_final.append(mejor_variable)
        else:
            var_interac_final.append(mejor_variable)
        dif_metrica = metrica_inicial - metrica_actual
        variables = [x for x in variables if x != mejor_variable]
        metrica_inicial = metrica_actual
        modelo_inicial = modelo

        if dif_metrica <= 5: #**Aumentando el numero soy mas exigente para la entrada
            # Si la métrica cambia, imprimir el paso y las variables actuales.
            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
        else:
            formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
            print('--------------------------------------- Step Forward: Entra ' + str(mejor_variable))
            print('')
            print('AIC = ' + str(metrica_inicial))
            print('')
            print('y ~ ' + formula)
            print('')
            
        # Etapa de retroceso (salida de variable)
        if ((dif_metrica > 5) and (len(var_cont_final + var_categ_final + var_interac_final) > 1)): #dif_metrica > 0 esta condición debe ser la contraria a **
            variables_probar = []
            metricas = []
            for x in var_cont_final + var_categ_final + var_interac_final:
                var_cont_probar, var_categ_probar, var_interac_probar = var_cont_final, var_categ_final, var_interac_final
                if x in var_cont:
                    var_cont_probar = [v for v in var_cont_probar if v != x]
                elif x in var_categ:
                    var_categ_probar = [v for v in var_categ_probar if v != x]
                else:
                    var_interac_probar = [v for v in var_interac_probar if v != x]
                # Ajustar un modelo sin la variable actual y calcular la métrica.
                modelo = glm(varObjBin, datos, var_cont_probar, var_categ_probar, var_interac_probar)['Modelo']
                variables_probar.append(x)
                metricas.append(calcular_metrica(modelo, datos, varObjBin, var_cont_probar,
                                                 var_categ_probar, var_interac_probar))
            # Imprimir las métricas para las variables probadas en la etapa de retroceso.
            print(pd.DataFrame({
                'Variable': [' - ' + str(x) for x in variables_probar], metodo: metricas
                }).sort_values(metodo, ascending=False).to_string(index=False))
            print('')
            mejor_variable = variables_probar[min(enumerate(metricas), key=lambda x: x[1])[0]]
            metrica_actual = metricas[min(enumerate(metricas), key=lambda x: x[1])[0]]

            # Actualizar las variables seleccionadas si la métrica mejora.
            if mejor_variable in var_cont:
                var_cont_final = [x for x in var_cont_final if x != mejor_variable]
            elif mejor_variable in var_categ:
                var_categ_final = [x for x in var_categ_final if x != mejor_variable]
            else:
                var_interac_final = [x for x in var_interac_final if x != mejor_variable]
            dif_metrica_salida = metrica_actual - metrica_inicial
            variables = [x for x in variables if x != mejor_variable]

            if dif_metrica_salida < 5: # aumentando el numero soy menos exigente para sacar
# =============================================================================
#                 # Si la métrica mejora, imprimir el paso y las variables actuales.
#                 if mejor_variable in var_cont:
#                     var_cont_final = [x for x in var_cont_final if x != mejor_variable]
#                 elif mejor_variable in var_categ:
#                     var_categ_final = [x for x in var_categ_final if x != mejor_variable]
#                 else:
#                     var_interac_final = [x for x in var_interac_final if x != mejor_variable]
# =============================================================================
                modelo_inicial = modelo
                metrica_inicial = metrica_actual																					   								
                formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
                print('--------------------------------------- Step Backward: Sale ' + str(mejor_variable))
                print('')
                print('AIC = ' + str(metrica_inicial))
                print('')
                print('y ~ ' + formula)
                print('')
            else:
                # Si no se elimina ninguna variable, simplemente imprimir el paso.
                if mejor_variable in var_cont:
                    var_cont_final.append(mejor_variable)
                elif mejor_variable in var_categ:
                    var_categ_final.append(mejor_variable)
                else:
                    var_interac_final.append(mejor_variable)
                formula = ' + '.join(var_cont_final + var_categ_final + ['*'.join(x) for x in var_interac_final])
                print('--------------------------------------- Step Backward: No sale ninguna')
                print('')
                print('AIC = ' + str(metrica_inicial))
                print('')
                print('y ~ ' + formula)
                print('')

    # Devolver el modelo final con el conjunto de variables seleccionado.
    return glm(varObjBin, datos, var_cont_final, var_categ_final, var_interac_final)


