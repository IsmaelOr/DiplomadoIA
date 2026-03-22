import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

def preprocess_dataset(df, target_col=None):
    """
    Aplica el flujo de preprocesamiento descrito en el diagrama:
    1. Filtrado por clase (se asume que target_col indica la columna objetivo).
    2. Eliminación de patrones redundantes (duplicados exactos).
    3. Eliminación de patrones indisociables (mismas features pero diferente clase).
    4. Imputación de valores perdidos: media para numéricos, moda para categóricos.
    5. Codificación de variables categóricas con Label Encoding.
    
    Parámetros:
    - df: DataFrame de pandas con los datos.
    - target_col: nombre de la columna que contiene la clase (opcional, pero necesaria para detectar patrones indisociables).
    
    Retorna:
    - DataFrame preprocesado (sin valores perdidos y con las categoricals codificadas).
    """
    
    data = df.copy()
    
    # --- 1. Filtramos por clase ---
    if target_col is not None and target_col not in data.columns:
        raise ValueError(f"La columna '{target_col}' no existe en el DataFrame.")
    
    # --- 2. Patrones redundantes (duplicados exactos) ---
    if data.duplicated().any():
        print("Se encontraron filas redundantes (duplicados exactos). Eliminando la primera instancia de cada grupo...")
        data = data.drop_duplicates(keep='first')
    
    # --- 3. Patrones indisociables (mismas features, diferente clase) ---
    if target_col is not None:
        features = [col for col in data.columns if col != target_col]
        grupos = data.groupby(features)[target_col].nunique()
        inconsistentes = grupos[grupos > 1].index  
        
        if len(inconsistentes) > 0:
            print("Se encontraron patrones indisociables (mismas features, clases distintas). Eliminando todas las instancias conflictivas...")
            idx_inconsistentes = data.set_index(features).index.isin(inconsistentes)
            data = data[~idx_inconsistentes]
    
    # --- 4. Detectar atributos con valores perdidos ---
    missing_cols = data.columns[data.isnull().any()].tolist()
    
    if missing_cols:
        print(f"Atributos con valores perdidos: {missing_cols}")
        for col in missing_cols:
            # Verificar si la columna es numérica
            if pd.api.types.is_numeric_dtype(data[col]):
                # Usar la media para imputar
                media = data[col].mean()
                data[col].fillna(media, inplace=True)
                print(f"  {col}: imputado con media ({media:.2f})")
            else:
                # Usar la moda para imputar
                moda = data[col].mode()
                if len(moda) > 0:
                    valor_moda = moda[0]
                    data[col].fillna(valor_moda, inplace=True)
                    print(f"  {col}: imputado con moda ({valor_moda})")
                else:
                    data[col].fillna('desconocido', inplace=True)
                    print(f"  {col}: sin moda, se usó 'desconocido'")
    else:
        print("No se encontraron valores perdidos.")
    
    # --- 5. Codificación de variables categóricas (Label Encoding) ---
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col is not None and target_col in cat_cols:
        cat_cols.remove(target_col)
    
    if cat_cols:
        print(f"Aplicando Label Encoding a las columnas categóricas: {cat_cols}")
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col].astype(str))
    else:
        print("No hay columnas categóricas que codificar (o ya estaban codificadas).")
    
    # --- 6. Devolvemos el dataset sin valores perdidos ---
    if data.isnull().any().any():
        print("Advertencia: Aún existen valores perdidos después del preprocesamiento.")
    else:
        print("Dataset preprocesado correctamente.")
    
    return data


def preprocess_dataset_modified(df, target_col=None, ordinal_cols=None):
    """
    Aplica el flujo de preprocesamiento descrito en el diagrama:
    1. Filtrado por clase (se asume que target_col indica la columna objetivo).
    2. Eliminación de patrones redundantes (duplicados exactos).
    3. Eliminación de patrones indisociables (mismas features pero diferente clase).
    4. Imputación de valores perdidos: media para numéricos, moda para categóricos.
    5. Codificación de variables categóricas:
        * Nominales -> OneHotEncoder.
        * Ordinales -> OrdinalEncoder.
    
    Parámetros:
    - df: DataFrame de pandas con los datos.
    - target_col: nombre de la columna que contiene la clase (opcional, pero necesaria para detectar patrones indisociables).
    - ordinal_cols: nombre de las columnas categoricas ordinales.

    Retorna:
    - X (datos procesados, sin valores perdidos y con las categoricals codificadas).
    - y (Variable objetivo)

    """
    
    data = df.copy()
    
    # --- 1. Filtramos por clase ---
    if target_col is not None and target_col not in data.columns:
        raise ValueError(f"La columna '{target_col}' no existe en el DataFrame.")
    
    # --- 2. Patrones redundantes (duplicados exactos) ---
    if data.duplicated().any():
        print("Se encontraron filas redundantes (duplicados exactos). Eliminando la primera instancia de cada grupo...")
        data = data.drop_duplicates(keep='first')
    
    # --- 3. Patrones indisociables (mismas features, diferente clase) ---
    if target_col is not None:
        features = [col for col in data.columns if col != target_col]
        grupos = data.groupby(features)[target_col].nunique()
        inconsistentes = grupos[grupos > 1].index  
        
        if len(inconsistentes) > 0:
            print("Se encontraron patrones indisociables (mismas features, clases distintas). Eliminando todas las instancias conflictivas...")
            idx_inconsistentes = data.set_index(features).index.isin(inconsistentes)
            data = data[~idx_inconsistentes]
    
    # --- 4. Detectar atributos con valores perdidos ---
    missing_cols = data.columns[data.isnull().any()].tolist()
    
    if missing_cols:
        print(f"Atributos con valores perdidos: {missing_cols}")
        for col in missing_cols:
            # Verificar si la columna es numérica
            if pd.api.types.is_numeric_dtype(data[col]):
                # Usar la media para imputar
                media = data[col].mean()
                data[col] = data[col].fillna(media)
                print(f"  {col}: imputado con media ({media:.2f})")
            else:
                # Usar la moda para imputar
                moda = data[col].mode()
                if len(moda) > 0:
                    valor_moda = moda[0]
                    data[col] = data[col].fillna(valor_moda)
                    print(f"  {col}: imputado con moda ({valor_moda})")
                else:
                    data[col].fillna('desconocido', inplace=True)
                    print(f"  {col}: sin moda, se usó 'desconocido'")
    else:
        print("No se encontraron valores perdidos.")
    
    # Separar X y y
    if target_col is not None:
        X = data.drop(columns=[target_col])
        y = data[target_col]
    else:
        X = data
        y = None

    # --- 5. Codificación de variables categóricas (OrdinalEncoder u OneHotEncoder) ---
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Si el usuario no especifica cuáles son ordinales
    if ordinal_cols is None:
        ordinal_cols = []
    
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

    if categorical_cols:
        # Pipelines
        nominal_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        ordinal_pipeline = Pipeline([
            ("ordinal", OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("nom", nominal_pipeline, nominal_cols),
                ("ord", ordinal_pipeline, ordinal_cols)
            ],
            remainder="passthrough" 
        )

        X_array = preprocessor.fit_transform(X)

        feature_names = preprocessor.get_feature_names_out()

        X = pd.DataFrame(X_array, columns=feature_names, index=data.index)
    else:
        print("No hay columnas categóricas que codificar (o ya estaban codificadas).")
        X = X.copy()
    
    # --- 6. Eliminar columnas constantes ---
    if isinstance(X, pd.DataFrame):
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            print(f"Eliminando columnas constantes: {list(constant_cols)}")
            X = X.drop(columns=constant_cols)


    # --- 7. Devolvemos el dataset sin valores perdidos ---
    if data.isnull().any().any():
        print("Advertencia: Aún existen valores perdidos después del preprocesamiento.")
    else:
        print("Dataset preprocesado correctamente.")

        
    return X, y
