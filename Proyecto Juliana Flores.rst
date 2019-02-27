
.. code:: ipython3

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime as DT
    import io
    import statsmodels.api as sm
    import imblearn
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    import seaborn as sb

.. code:: ipython3

    dataid = pd.read_csv('BASE_ID.txt', delimiter= "\t")

.. code:: ipython3

    dataid['FECHA_NACIMIENTO'] = pd.to_datetime(dataid['FECHA_NACIMIENTO'], format="%Y%m%d", errors='coerce')
    #dataid['FECHA_NACIMIENTO'] == dataid['FECHA_NACIMIENTO'].apply('{:06}'.format)
    now = pd.Timestamp(DT.datetime.now())
    dataid['FECHA_NACIMIENTO'] = pd.to_datetime(dataid['FECHA_NACIMIENTO'], format='%Y%m%d')    # 1
    dataid['FECHA_NACIMIENTO'] = dataid['FECHA_NACIMIENTO'].where(dataid['FECHA_NACIMIENTO'] < now, dataid['FECHA_NACIMIENTO'] -  np.timedelta64(100, 'Y'))   # 2
    dataid['Edad'] = (now - dataid['FECHA_NACIMIENTO']).astype('<m8[Y]')    # 3
    #Rangos hasta 17-29-39-49-59-69-80-110 millones
    bins = [3, 18, 30, 40, 50, 60, 70, 80, 102]
    labels = ['17','29', '39', '49', '59', '69', '80', '110']
    dataid['Rango_edad'] = pd.cut(dataid['Edad'], bins=bins, labels=labels, include_lowest=True)


.. code:: ipython3

    datamov = pd.read_csv('BASE_MOVIMIENTOS.txt', delimiter= "\s{2,}", engine='python')
    datamov




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ID</th>
          <th>FECHA_INFORMACION</th>
          <th>SALDO_AHORROS</th>
          <th>SALDO_FONDOS</th>
          <th>SALDO_CREDITO1</th>
          <th>SALDO_CREDITO2</th>
          <th>SALDO_TARJETA</th>
          <th>MONTO_COMPRAS1</th>
          <th>MONTO_CAJERO1</th>
          <th>MONTO_COMPRAS2</th>
          <th>MONTO_CAJERO2</th>
          <th>MONTO_ABONOS_NOMINA</th>
          <th>INDICADOR_MORA</th>
          <th>SALDO_ACTIVO</th>
          <th>SALDO_PASIVO</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3</td>
          <td>01JUL2017:00:00:00</td>
          <td>3340.00</td>
          <td>0,00</td>
          <td>$ 876.047,06</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>876047,06</td>
          <td>3340</td>
        </tr>
        <tr>
          <th>1</th>
          <td>171</td>
          <td>01MAY2017:00:00:00</td>
          <td>1070130.00</td>
          <td>0,00</td>
          <td>$ 7.828.500,12</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>460000.0</td>
          <td>897220.0</td>
          <td>0</td>
          <td>7828500,12</td>
          <td>1070133,18</td>
        </tr>
        <tr>
          <th>2</th>
          <td>313</td>
          <td>01OCT2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>644</td>
          <td>01MAY2017:00:00:00</td>
          <td>2204.88</td>
          <td>0,00</td>
          <td>$ 8.863.156,75</td>
          <td>$ 0,00</td>
          <td>2153170.0</td>
          <td>359023.0</td>
          <td>0.0</td>
          <td>8000.0</td>
          <td>160000.0</td>
          <td>748000.0</td>
          <td>0</td>
          <td>11016329,89</td>
          <td>2204,88</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1445</td>
          <td>01DEC2016:00:00:00</td>
          <td>692926.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>606720.0</td>
          <td>0</td>
          <td>0</td>
          <td>692926,32</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1835</td>
          <td>01FEB2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>3719890.0</td>
          <td>47094.3</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>3719889,43</td>
          <td>0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2075</td>
          <td>01MAR2017:00:00:00</td>
          <td>11585.00</td>
          <td>0,00</td>
          <td>$ 14.139.326,58</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>13060.0</td>
          <td>940000.0</td>
          <td>948266.0</td>
          <td>0</td>
          <td>14139326,58</td>
          <td>11584,95</td>
        </tr>
        <tr>
          <th>7</th>
          <td>771</td>
          <td>01JUN2017:00:00:00</td>
          <td>1005450.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>2308040.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>2308042,53</td>
          <td>1005445,18</td>
        </tr>
        <tr>
          <th>8</th>
          <td>622</td>
          <td>01NOV2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>1648220.0</td>
          <td>208450.0</td>
          <td>400000.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>1648223,89</td>
          <td>0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>760</td>
          <td>01MAY2017:00:00:00</td>
          <td>437996.00</td>
          <td>0,00</td>
          <td>$ 3.690.669,97</td>
          <td>$ 0,00</td>
          <td>2682600.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1200000.0</td>
          <td>1186420.0</td>
          <td>0</td>
          <td>6373266,62</td>
          <td>437995,97</td>
        </tr>
        <tr>
          <th>10</th>
          <td>1113</td>
          <td>01NOV2017:00:00:00</td>
          <td>96987.30</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>52600.0</td>
          <td>870000.0</td>
          <td>1038930.0</td>
          <td>0</td>
          <td>0</td>
          <td>96987,26</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2232</td>
          <td>01DEC2016:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>892043.0</td>
          <td>349000.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>892043,39</td>
          <td>0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>756</td>
          <td>01DEC2017:00:00:00</td>
          <td>2781340.00</td>
          <td>3455959,02</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>6237302,51</td>
        </tr>
        <tr>
          <th>13</th>
          <td>2492</td>
          <td>01MAR2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>14</th>
          <td>818</td>
          <td>01OCT2017:00:00:00</td>
          <td>47071.00</td>
          <td>0,00</td>
          <td>$ 3.799.021,38</td>
          <td>$ 3.043.574,08</td>
          <td>2501680.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>9344277,83</td>
          <td>47070,97</td>
        </tr>
        <tr>
          <th>15</th>
          <td>2147</td>
          <td>01OCT2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>16</th>
          <td>39</td>
          <td>01DEC2016:00:00:00</td>
          <td>350769.00</td>
          <td>0,00</td>
          <td>$ 3.496.482,50</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>3496482,5</td>
          <td>350768,58</td>
        </tr>
        <tr>
          <th>17</th>
          <td>269</td>
          <td>01MAR2017:00:00:00</td>
          <td>225692.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>93795.0</td>
          <td>2900000.0</td>
          <td>3407330.0</td>
          <td>0</td>
          <td>0</td>
          <td>225692</td>
        </tr>
        <tr>
          <th>18</th>
          <td>67</td>
          <td>01APR2017:00:00:00</td>
          <td>62256.40</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>700000.0</td>
          <td>693999.0</td>
          <td>0</td>
          <td>0</td>
          <td>62256,37</td>
        </tr>
        <tr>
          <th>19</th>
          <td>2107</td>
          <td>01JAN2017:00:00:00</td>
          <td>320256.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>1146360.0</td>
          <td>0.0</td>
          <td>20000.0</td>
          <td>131000.0</td>
          <td>1600000.0</td>
          <td>1197110.0</td>
          <td>0</td>
          <td>1146362,57</td>
          <td>320255,87</td>
        </tr>
        <tr>
          <th>20</th>
          <td>189</td>
          <td>01JUN2017:00:00:00</td>
          <td>1777990.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>235000.0</td>
          <td>280000.0</td>
          <td>1469220.0</td>
          <td>0</td>
          <td>0</td>
          <td>1777994,9</td>
        </tr>
        <tr>
          <th>21</th>
          <td>2194</td>
          <td>01FEB2017:00:00:00</td>
          <td>278638.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1690000.0</td>
          <td>999385.0</td>
          <td>0</td>
          <td>0</td>
          <td>278637,99</td>
        </tr>
        <tr>
          <th>22</th>
          <td>2341</td>
          <td>01JUN2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 7.110.518,29</td>
          <td>$ 0,00</td>
          <td>1711220.0</td>
          <td>132162.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>8821735,81</td>
          <td>0</td>
        </tr>
        <tr>
          <th>23</th>
          <td>332</td>
          <td>01FEB2017:00:00:00</td>
          <td>20184.60</td>
          <td>0,00</td>
          <td>$ 5.299.386,72</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>5299386,72</td>
          <td>20184,6</td>
        </tr>
        <tr>
          <th>24</th>
          <td>230</td>
          <td>01OCT2017:00:00:00</td>
          <td>38240.00</td>
          <td>0,00</td>
          <td>$ 8.207.754,23</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>8207754,23</td>
          <td>38239,98</td>
        </tr>
        <tr>
          <th>25</th>
          <td>403</td>
          <td>01OCT2017:00:00:00</td>
          <td>179665.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>186000.0</td>
          <td>890000.0</td>
          <td>1090040.0</td>
          <td>0</td>
          <td>0</td>
          <td>179664,74</td>
        </tr>
        <tr>
          <th>26</th>
          <td>2263</td>
          <td>01OCT2017:00:00:00</td>
          <td>247230.00</td>
          <td>0,00</td>
          <td>$ 5.744.604,57</td>
          <td>$ 0,00</td>
          <td>238219.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1100000.0</td>
          <td>1446970.0</td>
          <td>0</td>
          <td>5982823,76</td>
          <td>247229,54</td>
        </tr>
        <tr>
          <th>27</th>
          <td>1839</td>
          <td>01SEP2017:00:00:00</td>
          <td>240944.00</td>
          <td>0,00</td>
          <td>$ 14.850.179,65</td>
          <td>$ 0,00</td>
          <td>2175570.0</td>
          <td>1114000.0</td>
          <td>110000.0</td>
          <td>45740.0</td>
          <td>740000.0</td>
          <td>1378160.0</td>
          <td>0</td>
          <td>17025752,11</td>
          <td>240944,33</td>
        </tr>
        <tr>
          <th>28</th>
          <td>361</td>
          <td>01FEB2017:00:00:00</td>
          <td>390535.00</td>
          <td>0,00</td>
          <td>$ 49.230,81</td>
          <td>$ 0,00</td>
          <td>510412.0</td>
          <td>313700.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>559642,54</td>
          <td>390534,63</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1192</td>
          <td>01OCT2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>1996920.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>1996917,51</td>
          <td>0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>31410</th>
          <td>206</td>
          <td>01AUG2017:00:00:00</td>
          <td>57596.60</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>4954460.0</td>
          <td>0.0</td>
          <td>600000.0</td>
          <td>0.0</td>
          <td>880000.0</td>
          <td>1187140.0</td>
          <td>0</td>
          <td>4954455,18</td>
          <td>57596,63</td>
        </tr>
        <tr>
          <th>31411</th>
          <td>1242</td>
          <td>01SEP2017:00:00:00</td>
          <td>4560.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>4560</td>
        </tr>
        <tr>
          <th>31412</th>
          <td>1540</td>
          <td>01SEP2017:00:00:00</td>
          <td>5921820.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>5921819,57</td>
        </tr>
        <tr>
          <th>31413</th>
          <td>1255</td>
          <td>01JUN2017:00:00:00</td>
          <td>794.53</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3250950.0</td>
          <td>0</td>
          <td>0</td>
          <td>794,53</td>
        </tr>
        <tr>
          <th>31414</th>
          <td>611</td>
          <td>01MAR2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>124260.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>124259,96</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31415</th>
          <td>2445</td>
          <td>01JUL2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>3880590.0</td>
          <td>0.0</td>
          <td>100000.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>3880585,1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31416</th>
          <td>263</td>
          <td>01DEC2017:00:00:00</td>
          <td>245799.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>245798,82</td>
        </tr>
        <tr>
          <th>31417</th>
          <td>344</td>
          <td>01JAN2017:00:00:00</td>
          <td>349775.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1200000.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>349774,52</td>
        </tr>
        <tr>
          <th>31418</th>
          <td>1207</td>
          <td>01JUN2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>7822920.0</td>
          <td>20970.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>7822918,28</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31419</th>
          <td>2291</td>
          <td>01DEC2017:00:00:00</td>
          <td>1484280.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1048890.0</td>
          <td>1700000.0</td>
          <td>2484800.0</td>
          <td>0</td>
          <td>0</td>
          <td>1484284,87</td>
        </tr>
        <tr>
          <th>31420</th>
          <td>1085</td>
          <td>01JUN2017:00:00:00</td>
          <td>308473.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>967381.0</td>
          <td>0.0</td>
          <td>210000.0</td>
          <td>0.0</td>
          <td>2250000.0</td>
          <td>1120880.0</td>
          <td>0</td>
          <td>967380,77</td>
          <td>308472,56</td>
        </tr>
        <tr>
          <th>31421</th>
          <td>1186</td>
          <td>01APR2017:00:00:00</td>
          <td>85983.90</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2000000.0</td>
          <td>2122950.0</td>
          <td>0</td>
          <td>0</td>
          <td>85983,94</td>
        </tr>
        <tr>
          <th>31422</th>
          <td>1391</td>
          <td>01JAN2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>6121810.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>6121805,16</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31423</th>
          <td>1142</td>
          <td>01MAY2017:00:00:00</td>
          <td>43820.60</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>17050.0</td>
          <td>2640000.0</td>
          <td>1836090.0</td>
          <td>0</td>
          <td>0</td>
          <td>43820,61</td>
        </tr>
        <tr>
          <th>31424</th>
          <td>2442</td>
          <td>01JUL2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>8288820.0</td>
          <td>0.0</td>
          <td>100000.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>8288820,32</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31425</th>
          <td>671</td>
          <td>01OCT2017:00:00:00</td>
          <td>1066.00</td>
          <td>0,00</td>
          <td>$ 3.055.603,43</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>3055603,43</td>
          <td>1066</td>
        </tr>
        <tr>
          <th>31426</th>
          <td>1885</td>
          <td>01OCT2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 6.991.647,84</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>6991647,84</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31427</th>
          <td>595</td>
          <td>01JUL2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31428</th>
          <td>487</td>
          <td>01DEC2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 337.129,95</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>337129,95</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31429</th>
          <td>1697</td>
          <td>01APR2017:00:00:00</td>
          <td>199766.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>620000.0</td>
          <td>649117.0</td>
          <td>0</td>
          <td>0</td>
          <td>199766,47</td>
        </tr>
        <tr>
          <th>31430</th>
          <td>1822</td>
          <td>01SEP2017:00:00:00</td>
          <td>50188.00</td>
          <td>0,00</td>
          <td>$ 5.113.557,49</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1</td>
          <td>5113557,49</td>
          <td>50188</td>
        </tr>
        <tr>
          <th>31431</th>
          <td>2418</td>
          <td>01JAN2017:00:00:00</td>
          <td>91379.40</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>260000.0</td>
          <td>1550000.0</td>
          <td>1815000.0</td>
          <td>0</td>
          <td>0</td>
          <td>91379,39</td>
        </tr>
        <tr>
          <th>31432</th>
          <td>774</td>
          <td>01NOV2017:00:00:00</td>
          <td>1873500.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>1873503,57</td>
        </tr>
        <tr>
          <th>31433</th>
          <td>1427</td>
          <td>01MAY2017:00:00:00</td>
          <td>6789.47</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>6789,47</td>
        </tr>
        <tr>
          <th>31434</th>
          <td>870</td>
          <td>01MAR2017:00:00:00</td>
          <td>521196.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>880000.0</td>
          <td>1180320.0</td>
          <td>0</td>
          <td>0</td>
          <td>521195,75</td>
        </tr>
        <tr>
          <th>31435</th>
          <td>1593</td>
          <td>01JUL2017:00:00:00</td>
          <td>0.00</td>
          <td>0,00</td>
          <td>$ 2.354.466,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>2354466</td>
          <td>0</td>
        </tr>
        <tr>
          <th>31436</th>
          <td>1637</td>
          <td>01DEC2017:00:00:00</td>
          <td>12257.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>520000.0</td>
          <td>989751.0</td>
          <td>0</td>
          <td>0</td>
          <td>12257,01</td>
        </tr>
        <tr>
          <th>31437</th>
          <td>1953</td>
          <td>01NOV2017:00:00:00</td>
          <td>163163.00</td>
          <td>0,00</td>
          <td>$ 14.053.674,00</td>
          <td>$ 0,00</td>
          <td>1745200.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>510000.0</td>
          <td>1528740.0</td>
          <td>0</td>
          <td>15798870,88</td>
          <td>163163,38</td>
        </tr>
        <tr>
          <th>31438</th>
          <td>2115</td>
          <td>01AUG2017:00:00:00</td>
          <td>2662.29</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>2662,29</td>
        </tr>
        <tr>
          <th>31439</th>
          <td>2018</td>
          <td>01FEB2017:00:00:00</td>
          <td>276936.00</td>
          <td>0,00</td>
          <td>$ 0,00</td>
          <td>$ 0,00</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1450000.0</td>
          <td>1334400.0</td>
          <td>0</td>
          <td>0</td>
          <td>276935,68</td>
        </tr>
      </tbody>
    </table>
    <p>31440 rows × 15 columns</p>
    </div>



.. code:: ipython3

    dataid['SEXO']=np.where(dataid['SEXO'] =='Hombre', 'HOMBRE', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='M', 'HOMBRE', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='Masc.', 'HOMBRE', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='masculino', 'HOMBRE', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='varón', 'HOMBRE', dataid['SEXO'])

.. code:: ipython3

    dataid['SEXO']=np.where(dataid['SEXO'] =='mujer', 'MUJER', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='femenino', 'MUJER', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='Mujer', 'MUJER', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='F', 'MUJER', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='MUJER', 'MUJER', dataid['SEXO'])
    dataid['SEXO']=np.where(dataid['SEXO'] =='FEMENINO', 'MUJER', dataid['SEXO'])

.. code:: ipython3

    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] =='otros', 'OTROS', dataid['SITUACION_LABORAL'])
    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] =='Contrato fijo', 'CONTRATO FIJO', dataid['SITUACION_LABORAL'])
    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] =='temporal     ', 'CONTRATO TEMPORAL', dataid['SITUACION_LABORAL'])
    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] ==' desconocido   ', 'SIN CLASIFICAR', dataid['SITUACION_LABORAL'])
    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] =='contrato autonomo.', 'CONTRATO AUTONOMO', dataid['SITUACION_LABORAL'])
    dataid['SITUACION_LABORAL']=np.where(dataid['SITUACION_LABORAL'] =='CONTRATO AUNTONOMO', 'CONTRATO AUTONOMO', dataid['SITUACION_LABORAL'])


.. code:: ipython3

    dataid['fuga']=np.where(dataid['fuga'] ==np.nan, '0', dataid['fuga'])
    dataid['fuga']=np.where(dataid['fuga'] =="nan", '0', dataid['fuga'])
    dataid['fuga']=np.where(dataid['fuga'] =="1.0", '1', dataid['fuga'])


.. code:: ipython3

    #dataid['fuga'].value_counts()


.. code:: ipython3

    #Tener en cuenta
    pd.crosstab(dataid.SITUACION_LABORAL,dataid.fuga).plot(kind='bar')
    plt.title('Fugados por situación laboral')
    plt.xlabel('Tipo de trabajo')
    plt.ylabel('Fugados')
    plt.savefig('Fugados por trabajo')



.. image:: output_9_0.png


.. code:: ipython3

    #No tiene mucha importancia
    table=pd.crosstab(dataid.ESTADO_CIVIL,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Estado civil vs fugados')
    plt.xlabel('Estado civil')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por estado civil')



.. image:: output_10_0.png


.. code:: ipython3

    #No da información importante
    table=pd.crosstab(dataid.SEXO,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Sexo vs fugados')
    plt.xlabel('Sexo')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por sexo')



.. image:: output_11_0.png


.. code:: ipython3

    #La mayoría de gente está entre los 25 y lo 70 por ahí.
    dataid.Edad.hist()
    plt.title('Histograma de las edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('hist_age')



.. image:: output_12_0.png


.. code:: ipython3

    datamov['SALDO_AHORROS'].min()




.. parsed-literal::

    -1738.72



.. code:: ipython3

    datamov['SALDO_AHORROS'].max()




.. parsed-literal::

    93298700.0



.. code:: ipython3

    pd.crosstab(dataid.Rango_edad,dataid.fuga).plot(kind='bar')
    plt.title('Fugados por rango edad')
    plt.xlabel('Rango edad')
    plt.ylabel('Fugados')
    plt.savefig('Fugados por edad')



.. image:: output_15_0.png


.. code:: ipython3

    table=pd.crosstab(dataid.Rango_edad,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Edad vs fugados')
    plt.xlabel('Edad')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por rango ')



.. image:: output_16_0.png


.. code:: ipython3

    datamov[datamov['ID']==48]['INDICADOR_MORA']




.. parsed-literal::

    556      1
    3605     0
    6648     1
    16278    1
    16862    0
    17753    1
    24092    0
    28808    0
    Name: INDICADOR_MORA, dtype: int64



.. code:: ipython3

    datamov['SALDO_PASIVO'] = [float(x.replace(',', '.')) for x in datamov['SALDO_PASIVO']]
    datamov['SALDO_ACTIVO'] = [float(x.replace(',', '.')) for x in datamov['SALDO_ACTIVO']]
    dataid['CLIENTE_CC'] = [float(x.replace(',', '.')) for x in dataid['CLIENTE_CC']]

.. code:: ipython3

    prom_pas=[0]*2500
    for i in range(0, 2500):
        prom_pas[i] = sum(datamov[datamov['ID'] == i+1]['SALDO_PASIVO'])/len(datamov[datamov['ID'] == i+1]['SALDO_PASIVO'])
    


.. code:: ipython3

    dataid['Prom_pas'] = prom_pas


.. code:: ipython3

    prom_ac=[0]*2500
    for i in range(0, 2500):
        prom_ac[i] = sum(datamov[datamov['ID'] == i+1]['SALDO_ACTIVO'])/len(datamov[datamov['ID'] == i+1]['SALDO_ACTIVO'])
    


.. code:: ipython3

    dataid['Prom_ac'] = prom_ac


.. code:: ipython3

    moroso=[0]*2500
    for i in range(0, 2500):
        if sum(datamov[datamov['ID'] == i+1]['INDICADOR_MORA']) != 0:
            moroso[i]=1
        else:
            moroso[i] =0
        

.. code:: ipython3

    dataid['Moroso'] = moroso


.. code:: ipython3

    #Todo moroso es fugado.
    table=pd.crosstab(dataid.Moroso,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Moroso vs fugados')
    plt.xlabel('Moroso')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por morosos ')



.. image:: output_25_0.png


.. code:: ipython3

    #dataid['Prom_ac'].max()
    bins = [0, 2000000, 4000000, 6000000, 8000000, 12000000]
    labels = ['2','4', '6', '8', '12']
    dataid['Rango_pas'] = pd.cut(dataid['Prom_pas'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    bins = [0, 4000000, 8000000, 12000000, 16000000, 24000000]
    labels = ['4','8', '12', '16', '24']
    dataid['Rango_ac'] = pd.cut(dataid['Prom_ac'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    dataid




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>CLIENTE_CC</th>
          <th>FECHA_ALTA</th>
          <th>FECHA_NACIMIENTO</th>
          <th>SEXO</th>
          <th>ESTADO_CIVIL</th>
          <th>SITUACION_LABORAL</th>
          <th>fuga</th>
          <th>MES_DE_FUGA</th>
          <th>Edad</th>
          <th>Rango_edad</th>
          <th>Prom_pas</th>
          <th>Prom_ac</th>
          <th>Moroso</th>
          <th>Rango_pas</th>
          <th>Rango_ac</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
          <td>sep301996</td>
          <td>1930-08-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>2.0</td>
          <td>88.0</td>
          <td>110</td>
          <td>5.620097e+03</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.0</td>
          <td>may011986</td>
          <td>1953-06-30</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>65.0</td>
          <td>69</td>
          <td>0.000000e+00</td>
          <td>5.551901e+06</td>
          <td>0</td>
          <td>2</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.0</td>
          <td>dic011994</td>
          <td>1954-07-21</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>1</td>
          <td>11.0</td>
          <td>64.0</td>
          <td>69</td>
          <td>3.340000e+03</td>
          <td>1.184444e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.0</td>
          <td>ago271997</td>
          <td>1939-05-03</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>79.0</td>
          <td>80</td>
          <td>6.206543e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>jul211997</td>
          <td>1931-10-13</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>87.0</td>
          <td>110</td>
          <td>4.198419e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>5</th>
          <td>6.0</td>
          <td>jun131997</td>
          <td>1941-05-27</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>11.0</td>
          <td>77.0</td>
          <td>80</td>
          <td>1.622952e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>6</th>
          <td>7.0</td>
          <td>ene141997</td>
          <td>1936-09-17</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>82.0</td>
          <td>110</td>
          <td>3.544918e+06</td>
          <td>9.355054e+06</td>
          <td>0</td>
          <td>4</td>
          <td>12</td>
        </tr>
        <tr>
          <th>7</th>
          <td>8.0</td>
          <td>sep121997</td>
          <td>1962-05-23</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>56.0</td>
          <td>59</td>
          <td>0.000000e+00</td>
          <td>1.304496e+07</td>
          <td>0</td>
          <td>2</td>
          <td>16</td>
        </tr>
        <tr>
          <th>8</th>
          <td>9.0</td>
          <td>dic071999</td>
          <td>1935-07-27</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>SIN CLASIFICAR</td>
          <td>0</td>
          <td>NaN</td>
          <td>83.0</td>
          <td>110</td>
          <td>2.639296e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>9</th>
          <td>10.0</td>
          <td>sep121997</td>
          <td>1937-03-13</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>81.0</td>
          <td>110</td>
          <td>4.790643e+06</td>
          <td>4.780346e+05</td>
          <td>0</td>
          <td>6</td>
          <td>4</td>
        </tr>
        <tr>
          <th>10</th>
          <td>11.0</td>
          <td>oct091996</td>
          <td>1955-11-08</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>63.0</td>
          <td>69</td>
          <td>1.869159e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>11</th>
          <td>12.0</td>
          <td>may211997</td>
          <td>1964-10-06</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>7.058090e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>12</th>
          <td>13.0</td>
          <td>may211997</td>
          <td>1952-06-08</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>66.0</td>
          <td>69</td>
          <td>1.931457e+05</td>
          <td>1.338524e+07</td>
          <td>0</td>
          <td>2</td>
          <td>16</td>
        </tr>
        <tr>
          <th>13</th>
          <td>14.0</td>
          <td>sep121997</td>
          <td>1934-05-08</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>84.0</td>
          <td>110</td>
          <td>8.717380e+03</td>
          <td>2.623282e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>14</th>
          <td>15.0</td>
          <td>feb261992</td>
          <td>1940-09-16</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>78.0</td>
          <td>80</td>
          <td>2.228366e+05</td>
          <td>2.939130e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>15</th>
          <td>16.0</td>
          <td>sep121997</td>
          <td>1961-04-23</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>2.605159e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>16</th>
          <td>17.0</td>
          <td>sep121997</td>
          <td>1945-09-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>73.0</td>
          <td>80</td>
          <td>3.212239e+04</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>17</th>
          <td>18.0</td>
          <td>ago011997</td>
          <td>1961-07-07</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>2.295565e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>18</th>
          <td>19.0</td>
          <td>sep121997</td>
          <td>1944-11-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>1.283112e+05</td>
          <td>1.096898e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>19</th>
          <td>20.0</td>
          <td>sep121997</td>
          <td>1944-07-20</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>1.064809e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>20</th>
          <td>21.0</td>
          <td>ago121997</td>
          <td>1943-02-07</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>76.0</td>
          <td>80</td>
          <td>1.449231e+00</td>
          <td>7.495453e+06</td>
          <td>0</td>
          <td>2</td>
          <td>8</td>
        </tr>
        <tr>
          <th>21</th>
          <td>22.0</td>
          <td>sep121997</td>
          <td>1940-02-16</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>79.0</td>
          <td>80</td>
          <td>4.222311e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>6</td>
          <td>4</td>
        </tr>
        <tr>
          <th>22</th>
          <td>23.0</td>
          <td>sep121997</td>
          <td>1937-10-06</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>81.0</td>
          <td>110</td>
          <td>7.975854e+04</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>23</th>
          <td>24.0</td>
          <td>jul181997</td>
          <td>1962-01-01</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>2.881065e+04</td>
          <td>4.883736e+06</td>
          <td>0</td>
          <td>2</td>
          <td>8</td>
        </tr>
        <tr>
          <th>24</th>
          <td>25.0</td>
          <td>jul261997</td>
          <td>1944-11-19</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>6.366500e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>25</th>
          <td>26.0</td>
          <td>ago081997</td>
          <td>1942-04-03</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>76.0</td>
          <td>80</td>
          <td>8.088041e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>26</th>
          <td>27.0</td>
          <td>ago221997</td>
          <td>1944-05-10</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>4.009980e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>27</th>
          <td>28.0</td>
          <td>nov251999</td>
          <td>1965-01-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>1.371850e+03</td>
          <td>4.965973e+06</td>
          <td>0</td>
          <td>2</td>
          <td>8</td>
        </tr>
        <tr>
          <th>28</th>
          <td>29.0</td>
          <td>jul231997</td>
          <td>1941-09-21</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>77.0</td>
          <td>80</td>
          <td>0.000000e+00</td>
          <td>2.115007e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>29</th>
          <td>30.0</td>
          <td>sep121997</td>
          <td>1945-01-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>1.379710e+03</td>
          <td>2.319192e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2470</th>
          <td>2471.0</td>
          <td>sep192016</td>
          <td>1976-08-31</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>42.0</td>
          <td>49</td>
          <td>0.000000e+00</td>
          <td>1.040418e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2471</th>
          <td>2472.0</td>
          <td>sep222016</td>
          <td>1961-04-17</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>4.039515e+04</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2472</th>
          <td>2473.0</td>
          <td>sep222016</td>
          <td>1991-07-04</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>27.0</td>
          <td>29</td>
          <td>4.835753e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2473</th>
          <td>2474.0</td>
          <td>sep262016</td>
          <td>1964-03-22</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>1.059971e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2474</th>
          <td>2475.0</td>
          <td>sep282016</td>
          <td>1975-03-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>43.0</td>
          <td>49</td>
          <td>1.986908e+05</td>
          <td>1.462308e+07</td>
          <td>0</td>
          <td>2</td>
          <td>16</td>
        </tr>
        <tr>
          <th>2475</th>
          <td>2476.0</td>
          <td>sep282016</td>
          <td>1989-01-07</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>30.0</td>
          <td>29</td>
          <td>1.545263e+05</td>
          <td>4.261539e+05</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2476</th>
          <td>2477.0</td>
          <td>sep282016</td>
          <td>1991-06-19</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>27.0</td>
          <td>29</td>
          <td>3.392647e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2477</th>
          <td>2478.0</td>
          <td>sep292016</td>
          <td>1980-02-23</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>39.0</td>
          <td>39</td>
          <td>1.737218e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2478</th>
          <td>2479.0</td>
          <td>sep292016</td>
          <td>1982-03-27</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>36.0</td>
          <td>39</td>
          <td>7.760290e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>8</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2479</th>
          <td>2480.0</td>
          <td>sep302016</td>
          <td>1987-04-14</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>10.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>1.156707e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2480</th>
          <td>2481.0</td>
          <td>oct062016</td>
          <td>1994-10-26</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>24.0</td>
          <td>29</td>
          <td>8.742285e+04</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2481</th>
          <td>2482.0</td>
          <td>oct112016</td>
          <td>2000-06-01</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>18.0</td>
          <td>17</td>
          <td>6.313962e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>8</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2482</th>
          <td>2483.0</td>
          <td>oct112016</td>
          <td>1956-09-15</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>11.0</td>
          <td>62.0</td>
          <td>69</td>
          <td>4.366481e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2483</th>
          <td>2484.0</td>
          <td>oct142016</td>
          <td>1980-10-16</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>38.0</td>
          <td>39</td>
          <td>7.933174e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>8</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2484</th>
          <td>2485.0</td>
          <td>oct242016</td>
          <td>1987-12-06</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>9.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>0.000000e+00</td>
          <td>1.227399e+07</td>
          <td>0</td>
          <td>2</td>
          <td>16</td>
        </tr>
        <tr>
          <th>2485</th>
          <td>2486.0</td>
          <td>nov022016</td>
          <td>1993-03-28</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>9.0</td>
          <td>25.0</td>
          <td>29</td>
          <td>9.410532e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2486</th>
          <td>2487.0</td>
          <td>nov112016</td>
          <td>1991-08-21</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>7.0</td>
          <td>27.0</td>
          <td>29</td>
          <td>1.497541e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2487</th>
          <td>2488.0</td>
          <td>nov152016</td>
          <td>1987-07-09</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>3.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>1.026427e+07</td>
          <td>1.033195e+07</td>
          <td>0</td>
          <td>12</td>
          <td>12</td>
        </tr>
        <tr>
          <th>2488</th>
          <td>2489.0</td>
          <td>nov182016</td>
          <td>1962-08-14</td>
          <td>MUJER</td>
          <td>DIVORCIADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>56.0</td>
          <td>59</td>
          <td>0.000000e+00</td>
          <td>2.691863e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2489</th>
          <td>2490.0</td>
          <td>nov232016</td>
          <td>1980-02-12</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>5.0</td>
          <td>39.0</td>
          <td>39</td>
          <td>5.233245e+04</td>
          <td>2.783837e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2490</th>
          <td>2491.0</td>
          <td>nov292016</td>
          <td>1960-03-30</td>
          <td>MUJER</td>
          <td>SEPARADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>58.0</td>
          <td>59</td>
          <td>1.495241e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2491</th>
          <td>2492.0</td>
          <td>nov302016</td>
          <td>1985-04-24</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO AUTONOMO</td>
          <td>1</td>
          <td>4.0</td>
          <td>33.0</td>
          <td>39</td>
          <td>4.342488e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2492</th>
          <td>2493.0</td>
          <td>dic052016</td>
          <td>1972-05-06</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>6.0</td>
          <td>46.0</td>
          <td>49</td>
          <td>3.319409e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2493</th>
          <td>2494.0</td>
          <td>dic052016</td>
          <td>1959-09-27</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>59.0</td>
          <td>59</td>
          <td>3.719219e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2494</th>
          <td>2495.0</td>
          <td>dic072016</td>
          <td>1964-06-20</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>1</td>
          <td>8.0</td>
          <td>54.0</td>
          <td>59</td>
          <td>1.681031e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2495</th>
          <td>2496.0</td>
          <td>dic142016</td>
          <td>1979-02-12</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>40.0</td>
          <td>39</td>
          <td>6.321411e+04</td>
          <td>2.006889e+06</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2496</th>
          <td>2497.0</td>
          <td>dic162016</td>
          <td>1988-04-24</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>30.0</td>
          <td>29</td>
          <td>3.415234e+05</td>
          <td>7.402440e+06</td>
          <td>0</td>
          <td>2</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2497</th>
          <td>2498.0</td>
          <td>dic162016</td>
          <td>1978-07-26</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>4.0</td>
          <td>40.0</td>
          <td>39</td>
          <td>2.743499e+05</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2498</th>
          <td>2499.0</td>
          <td>dic212016</td>
          <td>1995-06-30</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>3.0</td>
          <td>23.0</td>
          <td>29</td>
          <td>5.218271e+04</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2499</th>
          <td>2500.0</td>
          <td>dic262016</td>
          <td>1946-12-01</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>1</td>
          <td>2.0</td>
          <td>72.0</td>
          <td>80</td>
          <td>5.924530e+06</td>
          <td>0.000000e+00</td>
          <td>0</td>
          <td>6</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    <p>2500 rows × 15 columns</p>
    </div>



.. code:: ipython3

    #Importante
    table=pd.crosstab(dataid.Rango_pas,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Pasivos vs fugados')
    plt.xlabel('Pasivos')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por pasivos ')



.. image:: output_29_0.png


.. code:: ipython3

    #Importante
    table=pd.crosstab(dataid.Rango_ac,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Activos vs fugados')
    plt.xlabel('Activos')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por Activos ')



.. image:: output_30_0.png


.. code:: ipython3

    prom_comp1=[0]*2500
    for i in range(0, 2500):
        prom_comp1[i] = sum(datamov[datamov['ID'] == i+1]['MONTO_COMPRAS1'])/len(datamov[datamov['ID'] == i+1]['MONTO_COMPRAS1'])

.. code:: ipython3

    prom_caj1=[0]*2500
    for i in range(0, 2500):
        prom_caj1[i] = sum(datamov[datamov['ID'] == i+1]['MONTO_CAJERO1'])/len(datamov[datamov['ID'] == i+1]['MONTO_CAJERO1'])

.. code:: ipython3

    prom_comp2=[0]*2500
    for i in range(0, 2500):
        prom_comp2[i] = sum(datamov[datamov['ID'] == i+1]['MONTO_COMPRAS2'])/len(datamov[datamov['ID'] == i+1]['MONTO_COMPRAS2'])

.. code:: ipython3

    prom_caj2=[0]*2500
    for i in range(0, 2500):
        prom_caj2[i] = sum(datamov[datamov['ID'] == i+1]['MONTO_CAJERO2'])/len(datamov[datamov['ID'] == i+1]['MONTO_CAJERO2'])

.. code:: ipython3

    prom_nom=[0]*2500
    for i in range(0, 2500):
        prom_nom[i] = sum(datamov[datamov['ID'] == i+1]['MONTO_ABONOS_NOMINA'])/len(datamov[datamov['ID'] == i+1]['MONTO_ABONOS_NOMINA'])

.. code:: ipython3

    dataid['Prom_comp1'] = prom_comp1
    dataid['Prom_comp2'] = prom_comp2
    dataid['Prom_caj1'] = prom_caj1
    dataid['Prom_caj2'] = prom_caj2
    dataid['Prom_nom'] = prom_nom

.. code:: ipython3

    bins = [0, 1000000, 2000000, 3000000, 4000000, 5000000]
    labels = ['1','2', '3', '4', '5']
    dataid['Rango_comp1'] = pd.cut(dataid['Prom_comp1'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    bins = [0, 1000000, 2000000, 3000000, 4000000, 5000000]
    labels = ['1','2', '3', '4', '5']
    dataid['Rango_caj2'] = pd.cut(dataid['Prom_caj2'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    bins = [0, 1000000, 2000000, 3000000, 4000000]
    labels = ['1','2', '3', '4']
    dataid['Rango_comp2'] = pd.cut(dataid['Prom_comp2'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    bins = [0, 500000, 1000000, 1500000, 2000000]
    labels = ['0.5','1', '1.5', '2']
    dataid['Rango_caj1'] = pd.cut(dataid['Prom_caj1'], bins=bins, labels=labels, include_lowest=True)

.. code:: ipython3

    #Importante
    table=pd.crosstab(dataid.Rango_comp1,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Compras créd vs fugados')
    plt.xlabel('Compras crédito')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por compras créd ')



.. image:: output_41_0.png


.. code:: ipython3

    table=pd.crosstab(dataid.Rango_comp2,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Compras deb vs fugados')
    plt.xlabel('Compras débito')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por compras deb ')



.. image:: output_42_0.png


.. code:: ipython3

    table=pd.crosstab(dataid.Rango_caj1,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Cajero créd vs fugados')
    plt.xlabel('Cajero crédito')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por Cajero créd ')



.. image:: output_43_0.png


.. code:: ipython3

    table=pd.crosstab(dataid.Rango_caj2,dataid.fuga)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Cajero deb vs fugados')
    plt.xlabel('Cajero débito')
    plt.ylabel('Proporción fugados')
    plt.savefig('Fugados por cajero débito')



.. image:: output_44_0.png


.. code:: ipython3

    sb.boxplot(x='fuga', y='Edad', data=dataid, palette='hls')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1c21c688d0>




.. image:: output_45_1.png


.. code:: ipython3

    dataid




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>CLIENTE_CC</th>
          <th>FECHA_ALTA</th>
          <th>FECHA_NACIMIENTO</th>
          <th>SEXO</th>
          <th>ESTADO_CIVIL</th>
          <th>SITUACION_LABORAL</th>
          <th>fuga</th>
          <th>MES_DE_FUGA</th>
          <th>Edad</th>
          <th>Rango_edad</th>
          <th>...</th>
          <th>Rango_ac</th>
          <th>Prom_comp1</th>
          <th>Prom_comp2</th>
          <th>Prom_caj1</th>
          <th>Prom_caj2</th>
          <th>Prom_nom</th>
          <th>Rango_comp1</th>
          <th>Rango_caj2</th>
          <th>Rango_comp2</th>
          <th>Rango_caj1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
          <td>sep301996</td>
          <td>1930-08-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>2.0</td>
          <td>88.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.0</td>
          <td>may011986</td>
          <td>1953-06-30</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>65.0</td>
          <td>69</td>
          <td>...</td>
          <td>8</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.0</td>
          <td>dic011994</td>
          <td>1954-07-21</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>1</td>
          <td>11.0</td>
          <td>64.0</td>
          <td>69</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.0</td>
          <td>ago271997</td>
          <td>1939-05-03</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>79.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>441215.384615</td>
          <td>0.000000</td>
          <td>1.743846e+06</td>
          <td>1.205821e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>jul211997</td>
          <td>1931-10-13</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>87.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>5</th>
          <td>6.0</td>
          <td>jun131997</td>
          <td>1941-05-27</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>11.0</td>
          <td>77.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>1.327692e+06</td>
          <td>1.253602e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>6</th>
          <td>7.0</td>
          <td>ene141997</td>
          <td>1936-09-17</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>82.0</td>
          <td>110</td>
          <td>...</td>
          <td>12</td>
          <td>1.253846e+04</td>
          <td>784.615385</td>
          <td>0.000000</td>
          <td>1.627692e+06</td>
          <td>1.080945e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>7</th>
          <td>8.0</td>
          <td>sep121997</td>
          <td>1962-05-23</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>56.0</td>
          <td>59</td>
          <td>...</td>
          <td>16</td>
          <td>9.607692e+04</td>
          <td>0.000000</td>
          <td>16923.076923</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>8</th>
          <td>9.0</td>
          <td>dic071999</td>
          <td>1935-07-27</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>SIN CLASIFICAR</td>
          <td>0</td>
          <td>NaN</td>
          <td>83.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>9</th>
          <td>10.0</td>
          <td>sep121997</td>
          <td>1937-03-13</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>81.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>4.944223e+05</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>10</th>
          <td>11.0</td>
          <td>oct091996</td>
          <td>1955-11-08</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>63.0</td>
          <td>69</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>15307.692308</td>
          <td>0.000000</td>
          <td>2.126154e+06</td>
          <td>2.144148e+06</td>
          <td>1</td>
          <td>3</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>11</th>
          <td>12.0</td>
          <td>may211997</td>
          <td>1964-10-06</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>12</th>
          <td>13.0</td>
          <td>may211997</td>
          <td>1952-06-08</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>66.0</td>
          <td>69</td>
          <td>...</td>
          <td>16</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>4.633633e+05</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>13</th>
          <td>14.0</td>
          <td>sep121997</td>
          <td>1934-05-08</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>84.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>14</th>
          <td>15.0</td>
          <td>feb261992</td>
          <td>1940-09-16</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>78.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>2.686308e+04</td>
          <td>523.076923</td>
          <td>164615.384615</td>
          <td>1.031538e+06</td>
          <td>9.593614e+05</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>15</th>
          <td>16.0</td>
          <td>sep121997</td>
          <td>1961-04-23</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>9.846154e+05</td>
          <td>1.247349e+06</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>16</th>
          <td>17.0</td>
          <td>sep121997</td>
          <td>1945-09-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>73.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>3700.000000</td>
          <td>0.000000</td>
          <td>9.953846e+05</td>
          <td>1.541547e+06</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>17</th>
          <td>18.0</td>
          <td>ago011997</td>
          <td>1961-07-07</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>189115.923077</td>
          <td>0.000000</td>
          <td>1.035385e+06</td>
          <td>1.266397e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>18</th>
          <td>19.0</td>
          <td>sep121997</td>
          <td>1944-11-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>1.258462e+06</td>
          <td>1.444224e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>19</th>
          <td>20.0</td>
          <td>sep121997</td>
          <td>1944-07-20</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>1.494795e+06</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>20</th>
          <td>21.0</td>
          <td>ago121997</td>
          <td>1943-02-07</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>76.0</td>
          <td>80</td>
          <td>...</td>
          <td>8</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>21</th>
          <td>22.0</td>
          <td>sep121997</td>
          <td>1940-02-16</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>79.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>11807.692308</td>
          <td>0.000000</td>
          <td>1.230769e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>22</th>
          <td>23.0</td>
          <td>sep121997</td>
          <td>1937-10-06</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>81.0</td>
          <td>110</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>1.236633e+06</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>23</th>
          <td>24.0</td>
          <td>jul181997</td>
          <td>1962-01-01</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>...</td>
          <td>8</td>
          <td>2.571187e+05</td>
          <td>0.000000</td>
          <td>546153.846154</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>24</th>
          <td>25.0</td>
          <td>jul261997</td>
          <td>1944-11-19</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>25</th>
          <td>26.0</td>
          <td>ago081997</td>
          <td>1942-04-03</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>76.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>58367.307692</td>
          <td>0.000000</td>
          <td>1.270769e+06</td>
          <td>1.293451e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>26</th>
          <td>27.0</td>
          <td>ago221997</td>
          <td>1944-05-10</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>784.615385</td>
          <td>0.000000</td>
          <td>1.319231e+06</td>
          <td>1.554992e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>27</th>
          <td>28.0</td>
          <td>nov251999</td>
          <td>1965-01-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>...</td>
          <td>8</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>28</th>
          <td>29.0</td>
          <td>jul231997</td>
          <td>1941-09-21</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>77.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>29</th>
          <td>30.0</td>
          <td>sep121997</td>
          <td>1945-01-23</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>0</td>
          <td>NaN</td>
          <td>74.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2470</th>
          <td>2471.0</td>
          <td>sep192016</td>
          <td>1976-08-31</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>42.0</td>
          <td>49</td>
          <td>...</td>
          <td>4</td>
          <td>1.283422e+06</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2471</th>
          <td>2472.0</td>
          <td>sep222016</td>
          <td>1961-04-17</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>57.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>3953.384615</td>
          <td>0.000000</td>
          <td>1.756154e+06</td>
          <td>1.754830e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2472</th>
          <td>2473.0</td>
          <td>sep222016</td>
          <td>1991-07-04</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>27.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>49153.846154</td>
          <td>0.000000</td>
          <td>1.621538e+06</td>
          <td>1.615355e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2473</th>
          <td>2474.0</td>
          <td>sep262016</td>
          <td>1964-03-22</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>54.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>8803.076923</td>
          <td>0.000000</td>
          <td>1.749231e+06</td>
          <td>1.772395e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2474</th>
          <td>2475.0</td>
          <td>sep282016</td>
          <td>1975-03-18</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>43.0</td>
          <td>49</td>
          <td>...</td>
          <td>16</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2475</th>
          <td>2476.0</td>
          <td>sep282016</td>
          <td>1989-01-07</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>30.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>2.409277e+04</td>
          <td>0.000000</td>
          <td>146923.076923</td>
          <td>1.026154e+06</td>
          <td>9.180338e+05</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2476</th>
          <td>2477.0</td>
          <td>sep282016</td>
          <td>1991-06-19</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>27.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>137561.307692</td>
          <td>0.000000</td>
          <td>1.082308e+06</td>
          <td>1.334704e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2477</th>
          <td>2478.0</td>
          <td>sep292016</td>
          <td>1980-02-23</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>39.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>188710.615385</td>
          <td>0.000000</td>
          <td>3.500000e+05</td>
          <td>2.215338e+05</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2478</th>
          <td>2479.0</td>
          <td>sep292016</td>
          <td>1982-03-27</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>36.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>392.307692</td>
          <td>0.000000</td>
          <td>3.000000e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2479</th>
          <td>2480.0</td>
          <td>sep302016</td>
          <td>1987-04-14</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>10.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>70544.692308</td>
          <td>0.000000</td>
          <td>1.254615e+06</td>
          <td>1.290613e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2480</th>
          <td>2481.0</td>
          <td>oct062016</td>
          <td>1994-10-26</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>24.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>1.410769e+06</td>
          <td>1.377574e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2481</th>
          <td>2482.0</td>
          <td>oct112016</td>
          <td>2000-06-01</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>18.0</td>
          <td>17</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2482</th>
          <td>2483.0</td>
          <td>oct112016</td>
          <td>1956-09-15</td>
          <td>MUJER</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>11.0</td>
          <td>62.0</td>
          <td>69</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>307637.461538</td>
          <td>0.000000</td>
          <td>5.984615e+05</td>
          <td>9.095446e+05</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2483</th>
          <td>2484.0</td>
          <td>oct142016</td>
          <td>1980-10-16</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>38.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2484</th>
          <td>2485.0</td>
          <td>oct242016</td>
          <td>1987-12-06</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>9.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>...</td>
          <td>16</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2485</th>
          <td>2486.0</td>
          <td>nov022016</td>
          <td>1993-03-28</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>9.0</td>
          <td>25.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>367612.000000</td>
          <td>0.000000</td>
          <td>1.777778e+05</td>
          <td>5.340500e+05</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2486</th>
          <td>2487.0</td>
          <td>nov112016</td>
          <td>1991-08-21</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>7.0</td>
          <td>27.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>13122.307692</td>
          <td>0.000000</td>
          <td>1.031538e+06</td>
          <td>1.056236e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2487</th>
          <td>2488.0</td>
          <td>nov152016</td>
          <td>1987-07-09</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>CONTRATO TEMPORAL</td>
          <td>1</td>
          <td>3.0</td>
          <td>31.0</td>
          <td>39</td>
          <td>...</td>
          <td>12</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2488</th>
          <td>2489.0</td>
          <td>nov182016</td>
          <td>1962-08-14</td>
          <td>MUJER</td>
          <td>DIVORCIADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>56.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2489</th>
          <td>2490.0</td>
          <td>nov232016</td>
          <td>1980-02-12</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>1</td>
          <td>5.0</td>
          <td>39.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2490</th>
          <td>2491.0</td>
          <td>nov292016</td>
          <td>1960-03-30</td>
          <td>MUJER</td>
          <td>SEPARADO</td>
          <td>OTROS</td>
          <td>0</td>
          <td>NaN</td>
          <td>58.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>119401.692308</td>
          <td>0.000000</td>
          <td>6.069231e+05</td>
          <td>1.528278e+06</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2491</th>
          <td>2492.0</td>
          <td>nov302016</td>
          <td>1985-04-24</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO AUTONOMO</td>
          <td>1</td>
          <td>4.0</td>
          <td>33.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>328800.000000</td>
          <td>0.000000</td>
          <td>3.615385e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2492</th>
          <td>2493.0</td>
          <td>dic052016</td>
          <td>1972-05-06</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>6.0</td>
          <td>46.0</td>
          <td>49</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>8461.538462</td>
          <td>0.000000</td>
          <td>4.338462e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2493</th>
          <td>2494.0</td>
          <td>dic052016</td>
          <td>1959-09-27</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>59.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>3846.153846</td>
          <td>0.000000</td>
          <td>1.473846e+06</td>
          <td>2.006744e+06</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2494</th>
          <td>2495.0</td>
          <td>dic072016</td>
          <td>1964-06-20</td>
          <td>HOMBRE</td>
          <td>UNION LIBRE</td>
          <td>OTROS</td>
          <td>1</td>
          <td>8.0</td>
          <td>54.0</td>
          <td>59</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>7.492308e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2495</th>
          <td>2496.0</td>
          <td>dic142016</td>
          <td>1979-02-12</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO FIJO</td>
          <td>0</td>
          <td>NaN</td>
          <td>40.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>1.158486e+05</td>
          <td>3453.846154</td>
          <td>0.000000</td>
          <td>2.307692e+04</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2496</th>
          <td>2497.0</td>
          <td>dic162016</td>
          <td>1988-04-24</td>
          <td>MUJER</td>
          <td>UNION LIBRE</td>
          <td>CONTRATO TEMPORAL</td>
          <td>0</td>
          <td>NaN</td>
          <td>30.0</td>
          <td>29</td>
          <td>...</td>
          <td>8</td>
          <td>0.000000e+00</td>
          <td>8455.384615</td>
          <td>0.000000</td>
          <td>8.615385e+04</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2497</th>
          <td>2498.0</td>
          <td>dic162016</td>
          <td>1978-07-26</td>
          <td>MUJER</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>4.0</td>
          <td>40.0</td>
          <td>39</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>47723.153846</td>
          <td>0.000000</td>
          <td>1.846154e+05</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2498</th>
          <td>2499.0</td>
          <td>dic212016</td>
          <td>1995-06-30</td>
          <td>HOMBRE</td>
          <td>SOLTERO</td>
          <td>OTROS</td>
          <td>1</td>
          <td>3.0</td>
          <td>23.0</td>
          <td>29</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>2499</th>
          <td>2500.0</td>
          <td>dic262016</td>
          <td>1946-12-01</td>
          <td>HOMBRE</td>
          <td>CASADO</td>
          <td>CONTRATO AUTONOMO</td>
          <td>1</td>
          <td>2.0</td>
          <td>72.0</td>
          <td>80</td>
          <td>...</td>
          <td>4</td>
          <td>0.000000e+00</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000e+00</td>
          <td>0.000000e+00</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0.5</td>
        </tr>
      </tbody>
    </table>
    <p>2500 rows × 24 columns</p>
    </div>



.. code:: ipython3

    X = dataid.loc[:, ['Prom_pas', 'Prom_ac', 'Prom_comp1', 'Prom_comp2']]
    X2 = dataid.loc[:, dataid.columns != 'fuga']
    y = dataid.loc[:, dataid.columns == 'fuga']
    Y = y.astype(np.float)

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

.. code:: ipython3

    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)


.. parsed-literal::

    /Users/Nadie/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Nadie/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)




.. parsed-literal::

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)



.. code:: ipython3

    y_pred = LogReg.predict(X_test)

.. code:: ipython3

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrix




.. parsed-literal::

    array([[529,   0],
           [219,   2]])



.. code:: ipython3

    import statsmodels.api as sm
    logit_model=sm.Logit(np.asarray(Y), np.asarray(X))
    result = logit_model.fit()
    print(result.summary2())


.. parsed-literal::

    Optimization terminated successfully.
             Current function value: 0.517739
             Iterations 7
                             Results: Logit
    =================================================================
    Model:              Logit            Pseudo R-squared: 0.139     
    Dependent Variable: y                AIC:              2596.6950 
    Date:               2019-02-27 14:56 BIC:              2619.9912 
    No. Observations:   2500             Log-Likelihood:   -1294.3   
    Df Model:           3                LL-Null:          -1502.7   
    Df Residuals:       2496             LLR p-value:      5.4120e-90
    Converged:          1.0000           Scale:            1.0000    
    No. Iterations:     7.0000                                       
    -------------------------------------------------------------------
              Coef.    Std.Err.      z       P>|z|     [0.025    0.975]
    -------------------------------------------------------------------
    x1       -0.0000     0.0000    -9.3399   0.0000   -0.0000   -0.0000
    x2       -0.0000     0.0000   -13.2046   0.0000   -0.0000   -0.0000
    x3       -0.0000     0.0000    -4.1676   0.0000   -0.0000   -0.0000
    x4       -0.0000     0.0000    -6.2225   0.0000   -0.0000   -0.0000
    =================================================================
    


.. code:: ipython3

    np.exp(result.params)




.. parsed-literal::

    array([0.99999949, 0.99999981, 0.99999657, 0.99999631])


