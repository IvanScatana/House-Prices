import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")


def clean_data(X):
    X=X.copy()
    X['MasVnrType'] = X['MasVnrType'].fillna('None')
    X['FireplaceQu'] = X['FireplaceQu'].fillna('None')
    X['GarageQual'] = X['GarageQual'].fillna('None')
    X['GarageFinish'] = X['GarageFinish'].fillna('None')
    X['GarageType'] = X['GarageType'].fillna('None')
    X['GarageCond'] = X['GarageCond'].fillna('None')

    X['Alley'] = X['Alley'].fillna('None')
    X['PoolQC'] = X['PoolQC'].fillna('None')
    X['GarageType'] = X['GarageType'].fillna('None') 

    X['GarageArea'] = X['GarageArea'].fillna(0) 
    X['MasVnrArea'] = X['MasVnrArea'].fillna(0)

    X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['YearBuilt'])
    X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']
    X['RemAge'] = X['YrSold'] - X['YearRemodAdd']
    X['HouseAge'] = X['YrSold'] - X['YearBuilt']
    X['IsNew'] = (X['HouseAge'] <= 5).astype(int)
    X['IsOld'] = (X['HouseAge'] >= 70).astype(int)
    X['IsHistoric'] = (X['HouseAge'] >= 100).astype(int)
    X['TotalBath'] = (X['FullBath'] + (0.5 * X['HalfBath']) +  #добавлено
                      X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
    X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    X['TotalSF'] = (X['GrLivArea'] + 
                    X['TotalBsmtSF'].fillna(0) + 
                    X['GarageArea'].fillna(0) +
                    X['WoodDeckSF'] + 
                    X['OpenPorchSF'] + 
                    X['EnclosedPorch'] + 
                    X['3SsnPorch'] + 
                    X['ScreenPorch'])
    
    X['TotalBathrooms'] = X['FullBath'] + 0.5*X['HalfBath'] + X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X['TotalRooms'] = X['TotRmsAbvGrd'] + X['BedroomAbvGr']
    X['AreaPerRoom'] = X['GrLivArea'] / (X['TotRmsAbvGrd'] + 1)
    X['QualityScore'] = X['OverallQual'] * X['OverallCond']
    X['IsRenovated'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)
    X['YearsSinceRenovation'] = X['YrSold'] - X['YearRemodAdd']
    
    return X


def imputer_groupby_Neighborhood(X):
    X = X.copy()
    X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    return X




# Настройка страницы
st.set_page_config(page_title="🏠Предсказание стоимости дома", layout="wide")

# 1. Загрузка готового пайплайна (Препроцессор + Модель)
@st.cache_resource
def load_assets():
    # Рекомендуется сохранять именно обученный pipeline целиком
    return joblib.load('final_pipeline.pkl') 

pipeline = load_assets()

st.title("🔮 Предсказание стоимости недвижимости")

# 2. Боковая панель с информацией о данных
with st.sidebar:
    st.header("О модели")
    st.info("Ансамбль: Стекинг + LightGBM + CatBoost")
    
    # Можно вывести пример данных или описание колонок
    if st.checkbox("Показать структуру входных данных"):
        st.write("MSZoning, Street, Alley, LotShape, LandContour, Utilities, " \
        "LotConfig, LandSlope, Neighborhood, Condition1, Condition2 etc.")

# 3. Загрузка данных пользователем
uploaded_file = st.file_uploader("Загрузите CSV-файл для оценки", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    
    st.subheader("Предпросмотр загруженных данных")
    st.dataframe(input_df.head(5))

    if st.button("🚀 Рассчитать стоимость"):
        try:
            # 4. Предсказание
            # Пайплайн сам сделает transform и predict
            preds_log = pipeline.predict(input_df)
            
            # 5. Обратное преобразование (если были логи)
            preds_final = np.expm1(preds_log)
            
            # 6. Отображение результатов
            st.success("Готово!")
            
            # Сводные метрики по загруженному файлу
            m1, m2, m3 = st.columns(3)
            m1.metric("Средняя цена", f"${preds_final.mean():,.0f}")
            m2.metric("Минимальная", f"${preds_final.min():,.0f}")
            m3.metric("Максимальная", f"${preds_final.max():,.0f}")
            
            # Таблица с результатами
            result_df = input_df.copy()
            result_df['Predicted_Price'] = preds_final
            st.dataframe(result_df)
            
            # Скачивание
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Скачать результаты (CSV)", csv, "results.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")
            st.warning("Убедитесь, что все необходимые колонки присутствуют в файле.")
else:
    st.write("👈 Загрузите CSV файл в левой части экрана или в поле выше.")

# 7. Сведения из "тестового файла" (если нужно показать аналитику по обучению)
st.divider()
st.subheader("📊 Аналитика модели")
# Здесь можно вывести график важности признаков или гистограмму ошибок с валидации
st.caption("Модель обучена на данных о 1460 домах. R² на валидации: 0.92")