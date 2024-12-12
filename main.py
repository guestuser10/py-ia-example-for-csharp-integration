import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

def load_data():
    """Carga los datos de entrenamiento y prueba."""
    # Leer los datos
    X_full = pd.read_csv(r'C:\flutter_proj\aspy\train.csv', index_col='Id')
    X_test_full = pd.read_csv(r'C:\flutter_proj\aspy\test.csv', index_col='Id')

    # Remover filas con valores nulos en el objetivo
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full['SalePrice']
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # Mantener solo predictores numéricos
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    return X, y, X_test

def preprocess_data(X, X_test):
    """Divide y preprocesa los datos."""
    # Dividir en conjuntos de entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Identificar columnas con valores nulos
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    # Crear nuevas columnas para indicar valores nulos
    for col in cols_with_missing:
        X_train[col + '_was_missing'] = X_train[col].isnull()
        X_valid[col + '_was_missing'] = X_valid[col].isnull()
        X_test[col + '_was_missing'] = X_test[col].isnull()

    # Imputar valores nulos
    imputer = SimpleImputer()
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns, index=X_valid.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_valid, y_train, y_valid, X_test

def train_and_evaluate(X_train, X_valid, y_train, y_valid):
    """Entrena el modelo y evalúa el rendimiento."""
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Validación
    preds_valid = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds_valid)
    print(f"MAE: {mae}")
    return model

def make_predictions(model, X_test):
    """Genera predicciones en los datos de prueba."""
    preds_test = model.predict(X_test)
    return preds_test

def plot_predictions(output):
    """Crea una gráfica de las predicciones."""
    plt.figure(figsize=(10, 6))
    plt.plot(output['Id'], output['SalePrice'], label='SalePrice', color='blue', marker='o', linestyle='-')
    plt.xlabel('Id')
    plt.ylabel('SalePrice')
    plt.title('Predicciones del modelo')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Cargar datos
    X, y, X_test = load_data()

    # Preprocesar datos
    X_train, X_valid, y_train, y_valid, X_test_processed = preprocess_data(X, X_test)

    # Entrenar y evaluar el modelo
    model = train_and_evaluate(X_train, X_valid, y_train, y_valid)

    # Generar predicciones para el conjunto de prueba
    preds_test = make_predictions(model, X_test_processed)

    # Guardar predicciones
    output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
    print("Predicciones:")
    print(output.head())
    output.to_csv(r'C:\flutter_proj\aspy\submission.csv', index=False)
    print("Ver todas las predicciones en 'submission.csv'.")
    plot_predictions(output)

