import numpy as np
from pyGRNN import GRNN
import statsmodels.api as sm
import math
from scipy.optimize import curve_fit
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

# Função utilizada quando há tendencia E sazonalidade

def trend_and_sazon(x_train,x_fut,y):
    y_ = serie_estocastica(y)
    minimo = abs(min(y_))
    y_train = np.array([valor + minimo for valor in y_])
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    y_pred = AGRNN.predict(x_fut)
    return convert_prediction_estocastica(y[-1],[valor - minimo for valor in y_pred])

# Função utilizada quando há tendencia mas não sazonalidade

def trend_and_not_sazon(x_train,x_fut,y,sem_trend,popt):
    minimo = abs(min(sem_trend))
    y_train = np.array([valor + minimo for valor in sem_trend])
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    y_pred = AGRNN.predict(x_fut)
    return convert_prediction_log(y,[valor - minimo for valor in y_pred],popt)

# Função utilizada quando não há tendencia

def not_trend(x_train,x_fut,y):
    y_train = np.array(y)
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    return AGRNN.predict(x_fut)


# Função que verifica se uma série precisa ser redimensionada de diária para semanal

def verify_vibration(serie):
    est = np.array(serie_estocastica(preprocessing.minmax_scale(serie)))
    vib = est.std()
    return vib > 0.1

# Função que verifica se existe sazonalidade em uma série

def autocorrelation(values,intervalo):
    if 28 < intervalo < 32:
        acorr = sm.tsa.acf(values, nlags = 12)
        auto = [acorr[12]]
    elif 6 < intervalo < 8:
        acorr = sm.tsa.acf(values, nlags = 52)
        auto = [acorr[4],acorr[52]]
    elif intervalo > 32:
        auto = [0]
    else:
        ano = round(365/intervalo)
        mes = round(31/intervalo)
        acorr = sm.tsa.acf(values, nlags = ano)
        auto = [acorr[mes],acorr[ano]]
    auto = [abs(valor) for valor in auto]
    print('Seasonality test result:',max(auto))
    return max(auto)

# Função que transforma uma série em estocástica

def serie_estocastica(values):
    values = list(values)
    new_values = [0]
    for i in range(1,len(values)):
        new_values.append(values[i] - values[i - 1])
    return new_values

# Função que converte uma predição estocástica em uma predição normal

def convert_prediction_estocastica(last_value,predictions):
    predictions = list(predictions)
    new_pred = [last_value]
    for i in range(len(predictions)):
        new_pred.append(new_pred[-1] + predictions[i])
    return new_pred[1:]

# Função que adiciona sazonalidade a uma predição que teve a sazonalidade removida

def convert_prediction_log(y,predictions,popt):
    len_y = len(y)
    len_p = len(predictions)
    new_pred = [predictions[i - len_y] + square(i,popt[0],popt[1],popt[2]) for i in range(len_y,len_y + len_p)]
    exp = [math.exp(valor) for valor in new_pred]
    estocastica = serie_estocastica(exp)
    return convert_prediction_estocastica(y[-1],estocastica)

# Função que identifica se um dia da semana é domingo

def monday(x):
    if x == 6:
        return 1
    return 0

# Função linear (utilizada para regressão)

def linear(x,a,b):
    return (a * x) + b

# Função quadrática (utilizada para regressão)

def square(x,a,b,c):
    return (a * (x ** 2)) + (b * x) + c

# Função que realiza uma regressão quadrática em um conjunto de dados

def fit(xdata,ydata,func):
    x = list(xdata)
    y = list(ydata)
    popt,pcov = curve_fit(func,x,y)
    return popt

# Função que verifica o intervalo médio entre observações de uma série temporal

def intervalo_medio(datas):
    datas.sort_values()
    intervalos = []
    for i in range(1,len(datas)):
        delta = datas[i] - datas[i - 1]
        intervalos.append(delta.days)
    return sum(intervalos) / len(intervalos)

# Função que cria datas futuras para gerar a previsão

def criar_tempo(periodos,intervalo,start):
    datas = [start]
    for i in range(periodos):
        if 28 < intervalo < 32:
            new_data = datas[-1] + relativedelta(months = 1)
        elif 6 < intervalo < 8:
            new_data = datas[-1] + timedelta(weeks = 1)
        elif 360 < intervalo < 370:
            new_data = datas[-1] + relativedelta(years = 1)
        else:
            new_data = datas[-1] + timedelta(days = intervalo)
        datas.append(new_data)
    return datas[1:]

# Função utilizada quando há tendencia E sazonalidade

def trend_and_sazon(x_train,x_fut,y):
    y_ = serie_estocastica(y)
    minimo = abs(min(y_))
    y_train = np.array([valor + minimo for valor in y_])
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    y_pred = AGRNN.predict(x_fut)
    return convert_prediction_estocastica(y[-1],[valor - minimo for valor in y_pred])

# Função utilizada quando há tendencia mas não sazonalidade

def trend_and_not_sazon(x_train,x_fut,y,sem_trend,popt):
    minimo = abs(min(sem_trend))
    y_train = np.array([valor + minimo for valor in sem_trend])
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    y_pred = AGRNN.predict(x_fut)
    return convert_prediction_log(y,[valor - minimo for valor in y_pred],popt)

# Função utilizada quando não há tendencia

def not_trend(x_train,x_fut,y):
    y_train = np.array(y)
    AGRNN = GRNN()
    AGRNN.fit(x_train, y_train)
    return AGRNN.predict(x_fut)

class Forecast:
    def __init__(self,datas,y,periodos,trend = None,seasonality = None,weekly = None):
        # Separação dos valores em arrays
        datas = pd.to_datetime(list(datas))
        y = list(y)

        # Verificar vibração
        if weekly == None:
            if verify_vibration(y):
                dataset = pd.DataFrame({'y':y},index = datas).resample('w').sum()
                y = list(dataset['y'])
                datas = dataset.index
                del(dataset)
                periodos = int(periodos / 7)
        else:
            if weekly:
                dataset = pd.DataFrame({'y':y},index = datas).resample('w').sum()
                y = list(dataset['y'])
                datas = dataset.index
                del(dataset)
                periodos = int(periodos / 7)

        # Criação do dataframe de treinamento

        df = pd.DataFrame({})
        df['1'] = [data.day for data in datas]
        df['2'] = [data.dayofweek for data in datas]
        df['3'] = [data.month for data in datas]
        df['4'] = [data.year for data in datas]
        df['5'] = [data.quarter for data in datas]
        df['6'] = [data.day_of_year for data in datas]
        df['7'] = [data.week for data in datas]
        df['8'] = [monday(data) for data in df['2']]
        df = df.reset_index()

        # Criação do dataframe futuro

        intervalo = round(intervalo_medio(datas))
        new_datas = criar_tempo(periodos,intervalo,datas[-1])
        predict_df = pd.DataFrame({})
        predict_df['index'] = list(range(len(df),len(df) + len(new_datas)))
        predict_df['1'] = [data.day for data in new_datas]
        predict_df['2'] = [data.dayofweek for data in new_datas]
        predict_df['3'] = [data.month for data in new_datas]
        predict_df['4'] = [data.year for data in new_datas]
        predict_df['5'] = [data.quarter for data in new_datas]
        predict_df['6'] = [data.day_of_year for data in new_datas]
        predict_df['7'] = [data.week for data in new_datas]
        predict_df['8'] = [monday(data) for data in predict_df['2']]

        # Detecção de trend

        if trend == None:
            corr = np.corrcoef(np.array(y),np.arange(0,len(y)))[0,1]
            print('Trend test result:',corr)
            trend = corr < -0.6 or corr > 0.6

        # Detecção de sazonalidade

        if trend:
            log = [math.log(valor) for valor in y]
            popt = fit(list(range(len(log))),log,square)
            sem_trend = [log[i] - square(i,popt[0],popt[1],popt[2]) for i in range(len(log))]
            if seasonality == None:
                seasonality = autocorrelation(sem_trend,intervalo) > 0.5

        # Preparação das variáveis para processamento

        last_index = len(df)
        final_df = pd.concat([df,predict_df]).reset_index(drop = True)
        # print(final_df.iloc[last_index - 5:last_index + 5])
        x_processado = preprocessing.minmax_scale(final_df.values)
        x_fut = x_processado[last_index:]
        x_train = x_processado[:last_index]

        # Treinamento e predição

        if trend and seasonality:
            print('Trend and seasonality detected')
            self.pred_trend = trend_and_sazon(x_train,x_fut,y)
        elif trend and not seasonality:
            print('Trend detected and seasonality not detected')
            self.pred_trend = trend_and_not_sazon(x_train,x_fut,y,sem_trend,popt)
        else:
            print('Trend not detected')
            self.pred_trend = not_trend(x_train,x_fut,y)

        self.train = pd.DataFrame({'data':datas,'y':y})
        self.prediction = pd.DataFrame({'data':new_datas,'y':self.pred_trend})
        self.result = pd.concat([self.train,self.prediction]).reset_index(drop = True)

    def plot_result(self,title = 'Fast Forecast'):
        fig,ax = plt.subplots(figsize = (8,6))
        plt.plot(self.train['data'],self.train['y'],label = 'data')
        plt.plot(self.prediction['data'],self.prediction['y'],label = 'prediction')
        plt.title(title)
        plt.legend()