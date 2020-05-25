import numpy as np
import pandas as pd
from MigracAnalyzer import UploadFromCSV
from SQLiteInspector import SQLiteInspector
import openpyxl
import re
import sqlite3

import plotly.graph_objects as go
import cufflinks as cf
import plotly.express as px
from urllib.request import urlopen
import json
import matplotlib.pyplot as plt
import networkx as nx
import vincent

def stackBar(fig, go, yearCol, df3, df4, okrug):
    fig = go.Figure(data=[
        go.Bar(name='Прибытие', x=yearCol[1:], y=df4[okrug].to_list()),
        go.Bar(name='Выбытие', x=yearCol[1:], y=df3[okrug].to_list())
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.update_layout(
    title={
        'text': "Внутренняя миграция населения по округам РФ ("+okrug+" ФО)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()

def custobBar(fig, go, yearCol, df3, df4, okrug):
    fig = go.Figure()
    ls = df3[okrug].to_list()
    for x in range(0,len(ls)):
        ls[x] *= -1
    fig.add_trace(go.Bar(x=yearCol[1:], y=df4[okrug].to_list(),
                base=(df4[okrug].to_list()*-1),
                marker_color='crimson',
                name='Прибытие'))
    fig.add_trace(go.Bar(x=yearCol[1:], y=ls,
                base=0,
                marker_color='lightslategrey',
                name='Выбытие'
                ))
    fig.update_layout(
    title={
        'text': "Внутренняя миграция населения по округам РФ ("+okrug+" ФО)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()

def main():
    cf.go_offline()
    cf.set_config_file(world_readable=True, theme='pearl', offline=True)
    cnx = sqlite3.connect('db_migraciya')
    db = SQLiteInspector("db_migraciya")

    ######################################################
    """ START Визуализация численности населения START """ 
    df = pd.read_sql_query("SELECT okrug,All2012 as \"2012\",All2013 as \"2013\",All2014 as \"2014\",All2015 as \"2015\""+
                           ",All2016 as \"2016\",All2017 as \"2017\",All2018 as \"2018\" FROM AllPeople", cnx)
    #print(df)
    cols = []
    for i, row in df.iterrows():
        cols.append(row['Okrug'])
        row['2012'] = row['2012'].replace(',','.')
        row['2012'] = str(float(row['2012'])*1000.0)
        row['2013'] = row['2013'].replace(',','.')
        row['2013'] = str(float(row['2013'])*1000.0)
        row['2014'] = row['2014'].replace(',','.')
        row['2014'] = str(float(row['2014'])*1000.0)
        row['2015'] = row['2015'].replace(',','.')
        row['2015'] = str(float(row['2015'])*1000.0)
        row['2016'] = row['2016'].replace(',','.')
        row['2016'] = str(float(row['2016'])*1000.0)
        row['2017'] = row['2017'].replace(',','.')
        row['2017'] = str(float(row['2017'])*1000.0)
        row['2018'] = row['2018'].replace(',','.')
        row['2018'] = str(float(row['2018'])*1000.0)
        
    df = df.drop(df.columns[0], axis='columns')
    df = pd.DataFrame(df.T)
    df.columns = cols
    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Численность населения по округам РФ", kind="bar")
    #fig.show()

    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Численность населения по округам РФ")
    #fig.show()
    """ FINAL Визуализация численности населения FINAL """
    ######################################################

    ######################################################
    """ START Визуализация уровень жизни START """ 
    df = pd.read_sql_query("SELECT okrug,\"2009\",\"2010\",\"2011\",\"2012\",\"2013\",\"2014\",\"2015\","+
                           "\"2016\",\"2017\",\"2018\",\"2019\" FROM LevelLife", cnx)
    cols = []
    for i, row in df.iterrows():
        cols.append(row['Okrug'])
        row['2009'] = row['2009'][:-2]
        row['2010'] = row['2010'][:-2]
        row['2011'] = row['2011'][:-2]
        row['2012'] = row['2012'][:-2]
        row['2013'] = row['2013'][:-2]
    df = df.drop(df.columns[0], axis='columns')
    df = pd.DataFrame(df.T)
    df.columns = cols
    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Уровень жизни по округам РФ", kind="bar")
    #fig.show()

    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Уровень жизни по округам РФ")
    #fig.show()
    """ FINAL Визуализация уровень жизни FINAL """
    
    """
    #Анализ уровень жизни
    for i, row in df.iterrows():
        minCell = int(row.min())
        maxCell = int(row.max())-minCell
        ls = list(map(int, row.to_list()))
        for x in range(0,len(ls)):
            ls[x]=round((ls[x]- minCell)/ maxCell,4)
        df.loc[i] = ls

    print(df.T)
    
    i=0
    columnsNamesNF = "id,"
    for col in df.T.columns.to_list():
        columnsNamesNF += "\""+col+"\","
        i += 1

    print(columnsNamesNF[:-1])

    #db.deleteValue("DELETE * FROM KlevelLife")

    

    i=1
    for index,row in df.T.iterrows():
        col = ""
        for j in list(map(str,row['2009':'2019'].to_list())):
            col += "\""+j+"\","
        db.insertValue("KlevelLife",columnsNamesNF[:-1],"\""+str(i)+"\","+col[:-1])
        i+=1
    """                                          

    ######################################################

    ######################################################
    """ START Визуализация загрязнения START """ 
    df = pd.read_sql_query("SELECT year,okrug,Q_PDK FROM Dirty  WHERE okrug LIKE \"Общее%\""+
        " UNION SELECT year,okrug,SUM(Q_PDK) FROM Dirty WHERE year = \"2017\" AND okrug != \"КФО\""+
                           "GROUP BY year,okrug", cnx)
    #print(df)    
    cols = []
    cols.append("Year")
    for i, row in df.iterrows():
        if i<=7:
            cols.append(row['Okrug'].replace("Общее по ",' '))

    df = df.drop(df.columns[1], axis='columns')
    rows =[]
    rowTmp = []
    j=0
    for i, row in df.iterrows():
        j+=1
        if j==1:
            rowTmp.append(row['year'])
        rowTmp.append(row['Q_PDK'])
        if j%8==0:
            j=0
            rows.append(rowTmp)
            rowTmp = []


    df1 = pd.DataFrame(np.array(rows), columns = cols)
    df1 = df1.drop(df1.columns[0], axis='columns')
    df1=df1.T
    df2 = pd.DataFrame(columns = ['okrug','2017','2018','2019','2020'])

    j=0
    for i, row in df1.iterrows():
        j+=1
        df2.loc[j-1]=[cols[j],row[0], row[1], row[2], row[3]]

    #df2 = df2.drop(df2.columns[0], axis='columns')
    df2 = df2.T
    df3 = pd.DataFrame(df2)
    df3.columns = cols[1:len(cols)]
    fig = df3.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Загрязнение по округам РФ", kind="bar")
    #fig.show()

    fig = df3.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Загрязнение по округам РФ")
    #fig.show()
    """ FINAL Визуализация загрязнения FINAL """
    
    """
    #Анализ загрязнения
    df3=df3.T.drop(df3.T.columns[0], axis='columns')
    df3=df3.T
    print(df3)
    
    df = df3.T
    df['2017']=df['2017'].astype('int')
    df['2018']=df['2018'].astype('int')
    df['2019']=df['2019'].astype('int')
    df['2020']=df['2020'].astype('int')
    df=df.T
    print(df)
    
    for i, row in df.iterrows():
        minCell = int(row.min())
        maxCell = int(row.max())-minCell
        print(minCell,maxCell)
        ls = list(map(int, row.to_list()))
        for x in range(0,len(ls)):
            ls[x]=round((ls[x]- minCell)/ maxCell,4)
        df.loc[i] = ls

    print(df.T)
    
    i=0
    columnsNamesNF = "id,"
    for col in df.T.columns.to_list():
        columnsNamesNF += "\""+col+"\","
        i += 1

    print(columnsNamesNF[:-1])

    for index,row in df.T.iterrows():
        col = ""
        for j in list(map(str,row['2017':'2020'].to_list())):
            col += "\""+j+"\","
        if index.find("ЦФО")!=-1:
            i=1
        elif index.find("СЗФО")!=-1:
            i=2
        elif index.find("ЮФО")!=-1:
            i=3
        elif index.find("СКФО")!=-1:
            i=4
        elif index.find("ПФО")!=-1:
            i=5
        elif index.find("УФО")!=-1:
            i=6
        elif index.find("СФО")!=-1:
            i=7
        elif index.find("ДФО")!=-1:
            i=8 
        db.insertValue("Kdirty",columnsNamesNF[:-1],"\""+str(i)+"\","+col[:-1])
    """    
    ######################################################

    ######################################################
    """ START Визуализация мирации выбытие START """ 
    df = pd.read_sql_query("SELECT * FROM Migraciya", cnx)
    cols = []
    dataTmp = []
    summ = 0
    df2 = pd.DataFrame(columns = ['year','okrug','sum'])
    for i, row in df.iterrows():
        summ = 0
        dataTmp = []
        if i<=7:
            cols.append(row['okrug'])
        dataTmp.append(row['year'])
        dataTmp.append(row['okrug'])
        if df.columns[2].find(row['okrug']) == -1:
            summ += row[df.columns[2]]
        if df.columns[3].find(row['okrug']) == -1:
            summ += row[df.columns[3]]
        if df.columns[4].find(row['okrug']) == -1:
            summ += row[df.columns[4]]
        if df.columns[5].find(row['okrug']) == -1:
            summ += row[df.columns[5]]
        if df.columns[6].find(row['okrug']) == -1:
            summ += row[df.columns[6]]
        if df.columns[7].find(row['okrug']) == -1:
            summ += row[df.columns[7]]
        if df.columns[8].find(row['okrug']) == -1:
            summ += row[df.columns[8]]
        if df.columns[9].find(row['okrug']) == -1:
            summ += row[df.columns[9]]
        dataTmp.append(summ)
        df2.loc[i]= dataTmp

    df4 = pd.read_sql_query("SELECT year FROM Migraciya GROUP BY year", cnx)
    yearCol = []
    yearCol.append("okrug")
    for i, row in df4.iterrows():
        yearCol.append(row['year'])

    df3 = pd.DataFrame(columns = yearCol)
    for i, row in df2.T.iterrows():
        x=0
        df3.loc[0]=[cols[0],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=1
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=2
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=3
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=4
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=5
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=6
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]
        x=7
        df3.loc[x]=[cols[x],row[0+x], row[8+x], row[16+x], row[24+x], row[32+x], row[40+x],
                            row[48+x], row[56+x], row[64+x], row[72+x], row[80+x], row[88+x],
                            row[96+x], row[104+x], row[112+x], row[120+x], row[128+x], row[136+x],row[144+x]]

    df3 = df3.drop(df3.columns[0], axis='columns')
    df3 = df3.T    
    df3.columns = cols
    
    fig = df3.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Внутренняя миграция населения по округам РФ(Выбытие)", kind="bar")
    #fig.show()

    fig = df3.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Внутренняя миграция населения по округам РФ(Выбытие)")
    #fig.show()
    """ FINAL Визуализация мирации выбытие FINAL """
    ######################################################


    ######################################################
    """ START Визуализация мирации прибытие START """ 
    dfCome = pd.read_sql_query("SELECT year,SUM(Центральный) as 'Центральный',SUM(\"Северо-Западный\") as 'Северо-Западный',"+
                           "SUM(Южный) as 'Южный',SUM(\"Северо-Кавказский\") as 'Северо-Кавказский',SUM(Приволжский) as 'Приволжский',"+
                           "SUM(Уральский) as 'Уральский',SUM(Сибирский) as 'Сибирский',SUM(Дальневосточный) as 'Дальневосточный' FROM Migraciya  GROUP BY year", cnx)
    for i, row in df.iterrows():
        for j, row2 in dfCome.iterrows():
            if str(row2['year']).find(row['year']) != -1:
                for row3 in dfCome.columns:
                    if row3.find(row['okrug']) != -1:                        
                        row2[row3] -= row[row['okrug']]
                        dfCome.loc[int(row2['year'][2:]),row3] -= row[row['okrug']]

    df4 = pd.DataFrame(columns = yearCol)
    dfCome = dfCome.drop(dfCome.columns[0], axis='columns')
    j=0
    for i, row in dfCome.T.iterrows():
        df4.loc[j]= [i, row[dfCome.T.columns[0]], row[dfCome.T.columns[1]], row[dfCome.T.columns[2]], row[dfCome.T.columns[3]],
                     row[dfCome.T.columns[4]], row[dfCome.T.columns[5]], row[dfCome.T.columns[6]], row[dfCome.T.columns[7]],
                     row[dfCome.T.columns[8]], row[dfCome.T.columns[9]], row[dfCome.T.columns[10]], row[dfCome.T.columns[11]],
                     row[dfCome.T.columns[12]], row[dfCome.T.columns[13]], row[dfCome.T.columns[14]], row[dfCome.T.columns[15]],
                     row[dfCome.T.columns[16]], row[dfCome.T.columns[17]], row[dfCome.T.columns[18]]]
        j+=1
    
    df4 = df4.drop(df4.columns[0], axis='columns')
    df4 = df4.T    
    df4.columns = cols
    
    fig = df4.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Внутренняя миграция населения по округам РФ(Прибытие)", kind="bar")
    #fig.show()

    fig = df4.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Внутренняя миграция населения по округам РФ(Прибытие)")
    #fig.show()

    fig = go.Figure(data=[
        go.Bar(name='Прибытие', x=yearCol[1:], y=df4['Центральный'].to_list()),
        go.Bar(name='Выбытие', x=yearCol[1:], y=df3['Центральный'].to_list())
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.update_layout(
    title={
        'text': "Внутренняя миграция населения по округам РФ (Центральный ФО)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    #fig.show()

    #stackBar(fig, go, yearCol, df3, df4, 'Северо-Западный')
    #stackBar(fig, go, yearCol, df3, df4, 'Южный')
    #stackBar(fig, go, yearCol, df3, df4, 'Северо-Кавказский')
    #stackBar(fig, go, yearCol, df3, df4, 'Приволжский')
    #stackBar(fig, go, yearCol, df3, df4, 'Уральский')
    #stackBar(fig, go, yearCol, df3, df4, 'Сибирский')
    #stackBar(fig, go, yearCol, df3, df4, 'Дальневосточный')

    #custobBar(fig, go, yearCol, df3, df4, 'Центральный')
    #custobBar(fig, go, yearCol, df3, df4, 'Северо-Западный')
    #custobBar(fig, go, yearCol, df3, df4, 'Южный')
    #custobBar(fig, go, yearCol, df3, df4, 'Северо-Кавказский')
    #custobBar(fig, go, yearCol, df3, df4, 'Приволжский')
    #custobBar(fig, go, yearCol, df3, df4, 'Уральский')
    #custobBar(fig, go, yearCol, df3, df4, 'Сибирский')
    #custobBar(fig, go, yearCol, df3, df4, 'Дальневосточный')

    dfCome = pd.read_sql_query("SELECT okrug,SUM(Центральный) as 'Центральный',"
    "SUM(\"Северо-Западный\") as 'Северо-Западный',"
    "SUM(Южный) as 'Южный',SUM(\"Северо-Кавказский\") as 'Северо-Кавказский',"
    "SUM(Приволжский) as 'Приволжский',"
    "SUM(Уральский) as 'Уральский',SUM(Сибирский) as 'Сибирский',"
    "SUM(Дальневосточный) as 'Дальневосточный' "
    "FROM Migraciya  GROUP BY okrug", cnx)

    
    dfCome = dfCome.set_index('okrug')
    
    feature_2=[]
    feature_1 = dfCome.index.to_list()
    feature_1 = feature_1 + feature_1 + feature_1 + feature_1 + feature_1 + feature_1 + feature_1 + feature_1
    feature_2 = feature_2+ feature_1
    feature_1.sort()    
    

    score = []
    df = pd.DataFrame({'f1': feature_1, 'f2': feature_2})

    for i, row in df.iterrows():
        score.append(dfCome.loc[row['f1']][row['f2']])

    df['score']=score

    G1 = nx.from_pandas_edgelist(df=df, source='f1', target='f2', edge_attr='score')
    G = nx.DiGraph(G1)
    
    
    pos = nx.spring_layout(G, k=10)  # For better example looking

    edges = G.edges()
    colors = [G[u][v]['score'] for u,v in edges]
    weights = [G[u][v]['score']/200000 for u,v in edges]
    
    nx.draw(G, pos, with_labels=True, width=weights)
    #labels = {e: G.edges[e]['score'] for e in G.edges}
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    #plt.show()
    
    """ FINAL Визуализация мирации прибытие FINAL """
    """
    #Анализ мирации
    print(df3)
    print(df4)
    df = df4-df3
    print(df)

    dfTmp = pd.read_sql_query("SELECT * FROM Okruga", cnx)
    print(dfTmp)
    
    for i, row in df.iterrows():
        minCell = int(row.min())
        maxCell = int(row.max())-minCell
        ls = list(map(int, row.to_list()))
        for x in range(0,len(ls)):
            ls[x]=round((ls[x]- minCell)/ maxCell,4)
        df.loc[i] = ls

    print(df.T)
    
    i=0
    columnsNamesNF = "id,"
    for col in df.T.columns.to_list():
        columnsNamesNF += "\""+col+"\","
        i += 1

    print(columnsNamesNF[:-1])
    
    #db.deleteValue("DELETE * FROM KlevelLife")

    for index,row in df.T.iterrows():
        for k,r in dfTmp.iterrows():
            if str(r['name']).find(str(index)) != -1:
                i=int(r['id'])
                break
        col = ""
        for j in list(map(str,row['2000':'2018'].to_list())):
            col += "\""+j+"\","        
        db.insertValue("Kmigr",columnsNamesNF[:-1],"\""+str(i)+"\","+col[:-1])
    """
    ######################################################


    ######################################################
    """ START Визуализация безработица START """ 
    df = pd.read_sql_query("SELECT okrug,SUM(\"2009\") as '2009',SUM(\"2010\") as '2010',SUM(\"2011\") as '2011',SUM(\"2012\") as '2012',"+
			 "SUM(\"2013\") as '2013',SUM(\"2014\") as '2014',SUM(\"2015\") as '2015',SUM(\"2016\") as '2016',"+
			 "SUM(\"2017\") as '2017',SUM(\"2018\") as '2018',SUM(\"2019\") as '2019',SUM(\"2020\") as '2020' FROM Unemployed GROUP BY okrug", cnx)
    cols = []
    cols.append("Year")
    
    for i, row in df.iterrows():
        cols.append(row['okrug'][:(row['okrug'].find("(до") if (row['okrug'].find("(до")!=-1) else len(row['okrug']))])#number if number >= 0 else -number

    df = df.drop(df.columns[0], axis='columns')
    df = df.T
    df.columns = cols[1:]
    
    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Безработица по округам РФ", kind="bar")
    #fig.show()

    fig = df.iplot(asFigure=True, xTitle="Год",
                    yTitle="Кол-во человек", title="Безработица по округам РФ")
    #fig.show()
    """ FINAL Визуализация безработица FINAL """
    
    """
    #Анализ Безработицы
    print(df.T)

    dfTmp = pd.read_sql_query("SELECT * FROM Okruga", cnx)
    print(dfTmp)
    
    for i, row in df.iterrows():
        minCell = int(row.min())
        maxCell = int(row.max())-minCell
        ls = list(map(int, row.to_list()))
        for x in range(0,len(ls)):
            ls[x]=round((ls[x]- minCell)/ maxCell,4)
        df.loc[i] = ls

    print(df.T)
    
    i=0
    columnsNamesNF = "id,"
    for col in df.T.columns.to_list():
        columnsNamesNF += "\""+col+"\","
        i += 1

    print(columnsNamesNF[:-1])

    #db.deleteValue("DELETE * FROM KlevelLife")

    for index,row in df.T.iterrows():
        for k,r in dfTmp.iterrows():
            if str(r['name']).find(str(index)) != -1:
                i=int(r['id'])
                break
        col = ""
        for j in list(map(str,row['2009':'2020'].to_list())):
            col += "\""+j+"\","
        db.insertValue("Kunemployed",columnsNamesNF[:-1],"\""+str(i)+"\","+col[:-1])
    """
    ######################################################

    

    ######################################################
    """ START Визуализация погоды START """ 
    df = pd.read_sql_query("SELECT t.id,n.name,n.lat,n.lan,year,"+
        "round((dec+jan+feb)/3) as 'winter', round((mar+apr+may)/3) as 'spring',round((jun+jul+aug)/3) as 'summer',round((sep+oct+nov)/3) as 'autumn',"+
        "round((dec+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov)/12) as 'ALL'"+
        " FROM NamesBases n "+
        " JOIN Temperature t ON t.id=n.id WHERE year BETWEEN '2000' AND '2018'", cnx)

    df = pd.read_sql_query("SELECT t.id,n.name,n.lat,n.lan,year,round((dec+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov)/12) as 'ALL'"
            "FROM Temperature t "
            "JOIN NamesBases n ON n.id=t.id "
            "WHERE n.name in ('Москва, ВДНХ','Архангельск','Ростов-на-Дону','Махачкала','Казань','Екатеринбург','Омск','Хабаровск') "
            "AND t.year BETWEEN '2000' AND '2018'", cnx)

    df2 = pd.DataFrame(columns = ['Okrug','2000','2001','2002','2003','2004','2005',
                            '2006','2007','2008','2009','2010','2011',
                            '2012','2013','2014','2015','2016','2017','2018'])
    #ls = ('Москва, ВДНХ','Архангельск','Ростов-на-Дону','Махачкала','Казань','Екатеринбург','Омск','Хабаровск')
    #df2['Okrug'] = ls
    

    ls = []
    name = ""
    based_name_okrug = {
        'Москва, ВДНХ': 'ЦФО (Москва, ВДНХ)',
        'Архангельск': 'СЗФО (Архангельск)',
        'Ростов-на-Дону': 'ЮФО (Ростов-на-Дону)',
        'Махачкала': 'СКФО (Махачкала)',
        'Казань': 'ПФО (Казань)',
        'Екатеринбург': 'УФО (Екатеринбург)',
        'Омск': 'СФО (Омск)',
        'Хабаровск': 'ДФО (Хабаровск)'
    }
    for index,row in df.iterrows():
        if (str(row['year']).find('2000')==-1 or int(index) == 0) and int(index) !=151:
            ls.append(row['ALL'])            
        else:
            if(int(index) == 151):
                ls.append(row['ALL']) 
            ls.insert(0,based_name_okrug[name])
            df2.loc[based_name_okrug[name]]=ls
            ls = []
            ls.append(row['ALL'])
        name = row['name']
        
            
    print(df2)
    df2 = df2.drop(df2.columns[0], axis='columns')
    df2 = df2.T
    fig = df2.iplot(asFigure=True, xTitle="Год",
                    yTitle="Градусы Цельсия", title="Погода по округам")
    fig.show()

    
    """ FINAL Визуализация погоды FINAL """
    ######################################################


    ######################################################
    """ START Визуализация K START """

    df1 = pd.read_sql_query('SELECT o.id,o.name,'
	   'm."2009"+l."2009"+(+u."2009") as \'2009\','
	   'm."2010"+l."2010"+(+u."2010") as \'2010\','
	   'm."2011"+l."2011"+(+u."2011") as \'2011\','
	   'm."2012"+l."2012"+(+u."2012") as \'2012\','
	   'm."2013"+l."2013"+(+u."2013") as \'2013\','
	   'm."2014"+l."2014"+(+u."2014") as \'2014\','
	   'm."2015"+l."2015"+(+u."2015") as \'2015\','
	   'm."2016"+l."2016"+(+u."2016") as \'2016\','
	   'm."2017"+l."2017"+(+u."2017") as \'2017\','
	   'm."2018"+l."2018"+(+u."2018") as \'2018\' '
            'FROM Okruga o'
            ' JOIN Kmigr m ON o.id=m.id'
            ' JOIN Kunemployed u ON o.id=u.id'
            ' JOIN KlevelLife l ON o.id=l.id'
            ' JOIN Kdirty d ON o.id=d.id'
            ' order by o.id', cnx)

    cols = df1['name'].to_list()
    df1 = df1.drop(df1.columns[0], axis='columns')
    df1 = df1.drop(df1.columns[0], axis='columns')
    df1 = df1.T    
    df1.columns = cols

    fig = df1.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Итог качества округов РФ", kind="bar")
    #fig.show()

    fig = df1.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Итог качества округов РФ")
    #fig.show()
    
    """ FINAL Визуализация K FINAL """

    """
    #Запись в базу итог Качества
    df1 = df1.T
    print(df1)
    i=0
    columnsNamesNF = "id,"
    for col in df1.columns.to_list():
        columnsNamesNF += "\""+col+"\","
        i += 1

    print(columnsNamesNF[:-1])
    
    dfTmp = pd.read_sql_query("SELECT * FROM Okruga", cnx)
    
    for index,row in df1.iterrows():
        for k,r in dfTmp.iterrows():
            if str(r['name']).find(str(index)) != -1:
                i=int(r['id'])
                break
        col = ""
        for j in list(map(str,row['2009':'2018'].to_list())):
            col += "\""+j+"\","
        db.insertValue("Kkachestvo",columnsNamesNF[:-1],"\""+str(i)+"\","+col[:-1])
    """   
    ######################################################

    ######################################################
    """ START Визуализация прогноза K START """

    df1 = df1.T
    df1.loc['Северо-Кавказский федеральный округ','2009'] += 0.9

    ls = []
    for index,row in df1.iterrows():
        ls.append((row['2010']-row['2009'])+(row['2011']-row['2010'])+
                  (row['2012']-row['2011'])+(row['2013']-row['2012'])+
                  (row['2014']-row['2013'])+(row['2015']-row['2014'])+
                  (row['2016']-row['2015'])+(row['2017']-row['2016'])+
                  (row['2018']-row['2017']))
    df1['Total']=ls

    fig = df1['Total'].T.iplot(asFigure=True, xTitle="Год",
                    yTitle="У.е.", title="Общий итог качества (миграция, уровень жизни, безработица)", kind="bar")
    #fig.show()

 
    """ FINAL Визуализация прогноза K FINAL """
    ######################################################

    df = pd.DataFrame(columns=['Okrug','Total'])
    j=0
    for row in df1['Total'].T:
        df.loc[j]=row
        j+=1

    
    df['Okrug']=df1['Total'].T.index.to_list()
    print(df)

    from urllib.request import urlopen
    with urlopen("file:///D:/Python/Projects/Parser%20Excel/map.geojson") as response:
    #with urlopen("file:///C:/Users/sds.TO/Desktop/%D0%9C%D0%98%D0%A4%D0%98/6%20%D1%81%D0%B5%D0%BC%D0%B5%D1%81%D1%82%D1%80/Python/Projects/Parser%20Excel/Regions.geojson") as response:
        counties = json.load(response)


   
    import plotly.express as px

    fig = px.choropleth_mapbox(df, geojson=counties, locations='Okrug', color='Total',
                               color_continuous_scale="Viridis",
                               range_color=(-0.5651, 0.3472),
                               mapbox_style="carto-positron",
                               zoom=3, center = {"lat": 41.41845703125, "lon":50.69471783819287},
                               opacity=0.5,
                               labels={'unemp':'unemployment rate'}
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    db.closeDB()
    
#*************************************************************************************************#    
def AllPeople():
    datas = UploadFromCSV(r"Datasets\AllPeople\allpeople.csv")
    datasTmp = datas.workAllPeopleFile();
    
    print(datasTmp.columns[1])

    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        if i != 0:
            columnsNames += col+" TEXT NOT NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])


    #db.createTable("AllPeople",""+ datasTmp.columns[0] + " TEXT PRIMARY KEY," + columnsNames[:-1])
    for index,row in datasTmp.iterrows():
        db.insertValue("AllPeople",columnsNamesNF[:-1],"\""+row[datasTmp.columns[0]]+"\","+
                                                       "\""+row[datasTmp.columns[1]]+"\","+
                                                       "\""+row[datasTmp.columns[2]]+"\","+
                                                       "\""+row[datasTmp.columns[3]]+"\","+
                                                       "\""+row[datasTmp.columns[4]]+"\","+
                                                       "\""+row[datasTmp.columns[5]]+"\","+
                                                       "\""+row[datasTmp.columns[6]]+"\","+
                                                       "\""+row[datasTmp.columns[7]]+"\","+
                                                       "\""+row[datasTmp.columns[8]]+"\","+
                                                       "\""+row[datasTmp.columns[9]]+"\","+
                                                       "\""+row[datasTmp.columns[10]]+"\","+
                                                       "\""+row[datasTmp.columns[11]]+"\","+
                                                       "\""+row[datasTmp.columns[12]]+"\","+
                                                       "\""+row[datasTmp.columns[13]]+"\","+
                                                       "\""+row[datasTmp.columns[14]]+"\","+
                                                       "\""+row[datasTmp.columns[15]]+"\","+
                                                       "\""+row[datasTmp.columns[16]]+"\","+
                                                       "\""+row[datasTmp.columns[17]]+"\","+
                                                       "\""+row[datasTmp.columns[18]]+"\","+
                                                       "\""+row[datasTmp.columns[19]]+"\","+
                                                       "\""+row[datasTmp.columns[20]]+"\","+
                                                       "\""+row[datasTmp.columns[21]]+"\"")


def Dirty():
    datas = UploadFromCSV(r"Datasets\Dirty\17.csv")
    datasTmp = datas.workDirty(',')
    
    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    db.dropTable("DROP TABLE Dirty")
    db.createTable("Dirty","id INT NOT NULL, year INT NOT NULL," + columnsNames[:-1] + ", PRIMARY KEY(id,year)")

    for index,row in datasTmp.iterrows():
        db.insertValue("Dirty","id,year,"+columnsNamesNF[:-1],""+str(index)+","+str(2017)+","+
                                                       "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\"")


    datas = UploadFromCSV(r"Datasets\Dirty\18.csv")
    datasTmp = datas.workDirty(',');
    
    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        
    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    for index,row in datasTmp.iterrows():
        db.insertValue("Dirty","id,year,"+columnsNamesNF[:-1],""+str(index)+","+str(2018)+","+
                                                       "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\"")
        
    datas = UploadFromCSV(r"Datasets\Dirty\19.csv")
    datasTmp = datas.workDirty(';');
    
    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        
    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    for index,row in datasTmp.iterrows():
        db.insertValue("Dirty","id,year,"+columnsNamesNF[:-1],""+str(index)+","+str(2019)+","+
                                                       "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\"")
    
    datas = UploadFromCSV(r"Datasets\Dirty\20.csv")
    datasTmp = datas.workDirty(',');
    
    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        
    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    for index,row in datasTmp.iterrows():
        db.insertValue("Dirty","id,year,"+columnsNamesNF[:-1],""+str(index)+","+str(2020)+","+
                                                       "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\"")



def LevelLife():
    datas = UploadFromCSV(r"Datasets\LevelLife\lvllife.csv")
    datasTmp = datas.workLvlLife(';');

    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += "\""+col+"\","
        if i != 0:
            columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])


    #db.createTable("LevelLife",""+ datasTmp.columns[0] + " TEXT PRIMARY KEY," + columnsNames[:-1])

    for index,row in datasTmp.iterrows():
        db.insertValue("LevelLife",columnsNamesNF[:-1],"\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\","+
                                                       "\""+str(row[datasTmp.columns[9]])+"\","+
                                                       "\""+str(row[datasTmp.columns[10]])+"\","+
                                                       "\""+str(row[datasTmp.columns[11]])+"\"")

def Unemployed():
    datas = UploadFromCSV(r"Datasets\Unemployed\unemployed.csv")
    datasTmp = datas.workUnemployed(';')

    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += "\""+col+"\","
        if i != 0:
            columnsNames += "\""+col+"\" TEXT NOT NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])


    #db.createTable("Unemployed",""+ datasTmp.columns[0] + " TEXT NOT NULL," + columnsNames[:-1] +", PRIMARY KEY(month,okrug)")
    for index,row in datasTmp.iterrows():
        db.insertValue("Unemployed",columnsNamesNF[:-1],"\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\","+
                                                       "\""+str(row[datasTmp.columns[9]])+"\","+
                                                       "\""+str(row[datasTmp.columns[10]])+"\","+
                                                       "\""+str(row[datasTmp.columns[11]])+"\","+
                                                       "\""+str(row[datasTmp.columns[12]])+"\","+
                                                       "\""+str(row[datasTmp.columns[13]])+"\"")

def Migraciya():
    #db.createTable("Migraciya","year TEXT NOT NULL, okrug TEXT NOT NULL,"+
    #               "\"Центральный\" INT,\"Северо-Западный\" INT,\"Южный\" INT,\"Северо-Кавказский\" INT,"+
    #               "\"Приволжский\" INT,\"Уральский\" INT,\"Сибирский\" INT,\"Дальневосточный\" INT ,PRIMARY KEY(year,okrug)")
    

    #wb = openpyxl.load_workbook(filename = 'Datasets\Migraciya\migr1.xlsx',read_only=True)
    wb = openpyxl.load_workbook(filename = 'Datasets\Migraciya\migr3.xlsx',read_only=True)
    #print(wb)
    sheet = wb['sh']
    #print(sheet)
    #print(sheet['B7'].value[:-2])
    #print(sheet['B7'].value)
    #print(re.match(r'([2][0-9]{3})г.', "Городские и сельские поселения") is not None)

    A = []
    X = []
    i=100
    yearTmp = ""
    for row in sheet.rows:
        #print(A)
        X = []
        for cell in row:
            #print("!" + str(cell.value))
            if((cell.value!=None) and (re.match(r'([2][0-9]{3}г.)', str(cell.value)) is not None)):#(str(cell.value).find("20") != -1)):
                print(str(cell.value))
                yearTmp = cell.value[:-2]
                i=0
            if(i>=5 and i<=12):
                if(len(X) <=9):
                    s = str(cell.value).replace(u'\xa0', u'')
                    X.append(s.replace(u' ', u''))

        if(i>=5 and i<=12):      
          X[1] = yearTmp
          A.append(X)
          db.insertValue("Migraciya","'okrug','year',"+
                                     "'Центральный','Северо-Западный','Южный','Северо-Кавказский',"+
                                     "'Приволжский','Уральский','Сибирский','Дальневосточный'","\""+X[0]+"\","+
                         "\""+X[1]+"\","+
                         "\""+X[2]+"\","+
                         "\""+X[3]+"\","+
                         "\""+X[4]+"\","+
                         "\""+X[5]+"\","+
                         "\""+X[6]+"\","+
                         "\""+X[7]+"\","+
                         "\""+X[8]+"\","+
                         "\""+X[9]+"\"")
          
        i+=1
    print(A)
    df = pd.DataFrame(np.array(A), columns=['okrug','year',
                                            'Центральный','Северо-Западный','Южный','Северо-Кавказский',
                                            'Приволжский','Уральский','Сибирский','Дальневосточный'])
    print(df)

def weatherData():
    """
    datas = UploadFromCSV(r"Datasets\Climat\SrRain\namesBase.csv")
    datasTmp = datas.workRain(';');
    print(datasTmp)

    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        if i == 0:
            columnsNames += col+" INT NOT NULL,"
        elif i==1:
            columnsNames += col+" TEXT NOT NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    #db.createTable("NamesBases",columnsNames +"PRIMARY KEY(id)")
    for index,row in datasTmp.iterrows():
        db.insertValue("NamesBases",columnsNamesNF[:-1],     "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\"")
    """
    #datas = UploadFromCSV(r"Datasets\Climat\SrRain\rain.csv")
    datas = UploadFromCSV(r"Datasets\Climat\SrTemperature\temperature.csv")
    datasTmp = datas.workRain(';');
    datasTmp = datasTmp.fillna("NULL")
    print(datasTmp)

    columnsNames = ""
    columnsNamesNF = ""
    i = 0
    for col in datasTmp.columns:
        columnsNamesNF += col +","
        if i <= 1:
            columnsNames += col+" INT NOT NULL,"
        elif i>1:
            columnsNames += col+" REAL NULL,"
        i += 1
        

    print(columnsNames[:-1])
    print(columnsNamesNF[:-1])

    #db.dropTable("DROP TABLE Rain")
    #db.createTable("Temperature",columnsNames +"FOREIGN KEY(id) REFERENCES NamesBases(id), PRIMARY KEY(id,year)")
    for index,row in datasTmp.iterrows():
        if(index >= 24795):
            try:
                db.insertValue("Temperature",columnsNamesNF[:-1],     "\""+str(row[datasTmp.columns[0]])+"\","+
                                                       "\""+str(row[datasTmp.columns[1]])+"\","+
                                                       "\""+str(row[datasTmp.columns[2]])+"\","+
                                                       "\""+str(row[datasTmp.columns[3]])+"\","+
                                                       "\""+str(row[datasTmp.columns[4]])+"\","+
                                                       "\""+str(row[datasTmp.columns[5]])+"\","+
                                                       "\""+str(row[datasTmp.columns[6]])+"\","+
                                                       "\""+str(row[datasTmp.columns[7]])+"\","+
                                                       "\""+str(row[datasTmp.columns[8]])+"\","+
                                                       "\""+str(row[datasTmp.columns[9]])+"\","+
                                                       "\""+str(row[datasTmp.columns[10]])+"\","+
                                                       "\""+str(row[datasTmp.columns[11]])+"\","+
                                                       "\""+str(row[datasTmp.columns[12]])+"\","+
                                                       "\""+str(row[datasTmp.columns[13]])+"\"")
            except Exception as e:
                print(index, e)


def Coords():
    #db.alterTable("ALTER TABLE NamesBases ADD COLUMN lat TEXT;")
    #db.alterTable("ALTER TABLE NamesBases ADD COLUMN lan TEXT;")

    datas = UploadFromCSV(r"Datasets\Climat\SrTemperature\coords.csv")
    datasTmp = datas.workCoords(';');

    print(datasTmp)

    for index,row in datasTmp.iterrows():
        s = row['широта'].replace('?','’')
        s1 = row['долгота'].replace('?','’')
        s = s.replace(' о', '°')
        s = s.replace('o', '°')
        s = s.replace(' ', '')
        s1 = s1.replace(' о', '°')
        s1 = s1.replace(' о ', '°')
        s1 = s1.replace(' ', '')
        s1 = s1.replace('o', '°')
        datasTmp.loc[index]=[row['Индекс'],row['Наименование станции'],s.replace(' o', '°'),s1.replace(' o', '°')]

    for index,row in datasTmp.iterrows():
        db.updateValue("UPDATE NamesBases SET lat = '"+row['широта']+"', lan = '"+row['долгота']+"' WHERE id = '"+str(row['Индекс'])+"'");
     
if __name__=="__main__":
    main()
