import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main(verbosity=False):
    st.header("You Do #1")
    #st.markdown(""" Ultimate loss function will be optimizing the parameters of both models at the same time """)
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target
    st.dataframe(X)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))
    st.dataframe(df)
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols", trendline_scope="overall")
    st.plotly_chart(fig, use_container_width=True)

#split data into training and testing sets
    X_train, X_test, y_train, y_test=train_test_split(df['MedInc'].to_numpy(), df['Price'].to_numpy(), test_size=0.25, random_state= 100)

#Linear Regression Model
    linear_reg = LinearRegression()

#fit/predict data
    linear_reg.fit(X_train.reshape(-1, 1), y_train)
    lr_prices = linear_reg.predict(X_test.reshape(-1, 1))

# Mean Square Error Calculation
    def mean_squared_error(act, pred):
        diff = pred - act
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()
        return mean_diff

# Root Mean Square Error Calculation
    def root_mean_squared_error(act, pred):
        diff = pred - act
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()
        rmse_val = np.sqrt(mean_diff)
        return rmse_val

# Mean Absolute Error
    def mean_absolute_error(act, pred):
        diff = pred - act
        abs_diff = np.absolute(diff)
        mean_diff = abs_diff.mean()
        return mean_diff

# custom huber loss function
    def huber_loss_error(y_true, y_pred, delta=0.1):
        res = []
        for i in zip(y_true, y_pred):
            if abs(i[0] - i[1]) <= delta:
                res.append(0.5 * ((i[0] - i[1]) ** 2))
            else:
                res.append(delta * ((abs(i[0] - i[1])) - 0.5 * (delta ** 2)))
        # can also be write as:
        # np.where(np.abs(y_true-y_pred) < delta, 0.5*(y_true-y_pred)**2 , delta*(np.abs(y_true-y_pred)-0.5*delta))
        return res  # np.sum(res)

    df_predict = pd.DataFrame({'MedInc': X_test, 'Actual Price': y_test, 'Linear Reg Predicted Price': linreg_prices })
    df_predict

    st.write('Mean Squared Error:', mean_squared_error(y_test, linreg_prices))
    st.write('Root Mean Squared Error:', root_mean_squared_error(y_test, linreg_prices))
    st.write('Mean Absolute Error:', mean_absolute_error(y_test, linreg_prices))
    st.write('huber loss error  :', huber_loss_error(y_test, linreg_prices, 1))
if __name__ == '__main__':
     main(st.sidebar.checkbox("verbosity"))
