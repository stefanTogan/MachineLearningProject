import csv
import sys

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def multipleK(lista):
    dfu = pd.DataFrame()
    useri = []
    for i, listuta1 in enumerate(lista):
        remove = 0
        for listuta2 in lista[i + 1:]:
            for item in listuta1:
                if item in listuta2:
                    useri.append(item)
    dfu['useri'] = useri
    dfu.drop_duplicates(inplace=True, ignore_index=True)
    print(dfu)

def userKudosCount(lista):
    useri = []
    kudos = []
    for listuta in lista:
        for user in listuta:
            useri.append(user)
            kudos.append(0)
    print('gata cu userii')
    for u, luser in enumerate(useri):
        for listuta in lista:
            if luser in listuta:
                kudos[u] = kudos[u] + 1
    print('gata si cu kudos')
    dfu = pd.DataFrame()
    dfu['useri'] = useri
    dfu['kudos'] = kudos
    dfu.drop_duplicates(inplace=True, ignore_index=True)
    print(dfu)
    dfu.to_csv('useri.csv')

def tabel_mic(useri,df):
    useri_small = useri.loc[useri['kudos'] > 50, :]
    useri_small.reset_index(inplace=True, drop=True)
    dfF = df.iloc[:, 0]
    dfU = df.iloc[:, 1]
    listaFicuri = []
    listaUseri = []
    listaKudos = []
    listaUseriSmall = []
    for i in dfF:
        listaFicuri.append(i)
    for i in dfU:
        listaUseri.append(ast.literal_eval(i))
    for i in useri_small.loc[:, 'useri']:
        listaUseriSmall.append(i)
    print(useri_small.__len__())
    # df = df.set_index('index')
    # df.groupby([df.index, 'kudos']).count().plot(kind='bar')

    for i, useri in enumerate(listaUseri):
        temp = []
        for user in listaUseriSmall:
            temp.append(1 if user in useri else 0)
        useri_small[str(listaFicuri[i])] = pd.Series(temp)

    useri_small.drop('kudos', axis=1, inplace=True)
    useri_small.drop('index', axis=1, inplace=True)
    sums= useri_small.sum(axis=0)
    cols = useri_small.columns
    print(sums)
    for col in cols[1:]:
       if sums[col] < 10:
           useri_small.drop(col,axis=1,inplace=True)
    useri_small.to_csv('kudos_tiny.csv', index=False)

def tabel_mare(useri,df):
    useri_big = useri
    dfF = df.iloc[:, 0]
    dfU = df.iloc[:, 1]
    listaFicuri = []
    listaUseri = []
    listaKudos = []
    listaUseriBig = []
    for i in dfF:
        listaFicuri.append(i)
    for i in dfU:
        listaUseri.append(ast.literal_eval(i))
    for i in useri_big.loc[:, 'useri']:
        listaUseriBig.append(i)
    print(useri_big.__len__())
    # df = df.set_index('index')
    # df.groupby([df.index, 'kudos']).count().plot(kind='bar')

    for i, useri in enumerate(listaUseri):
        temp = []
        for user in listaUseriBig:
            temp.append(1 if user in useri else 0)
        useri_big[str(listaFicuri[i])] = pd.Series(temp)

    useri_big.drop('kudos', axis=1, inplace=True)
    useri_big.drop('index', axis=1, inplace=True)

    useri_big.to_csv('kudos_all.csv', index=False)

def tagCount(lista):
    tags = []
    for listuta in lista:
        for tag in listuta:
            tags.append(tag)
    tags = list(dict.fromkeys(tags))
    nr = [0] * tags.__len__()
    for t, tag in enumerate(tags):
        for listuta in lista:
            if tag in listuta:
                nr[t] = nr[t] + 1

    n=tags.__len__()
    for i in range(n-1,-1,-1):
        if nr[i] < 20:
            tags.pop(i)
            nr.pop(i)
    return tags

def tabel_taguri(kudos):
    ficuri = kudos.columns[1:].tolist()
    ficuri = [eval(x) for x in ficuri]
    df = pd.read_csv('tags.csv', header=0)

    df = df.loc[df['id'].isin(ficuri), :]
    ficuri = df.loc[:,'id'].tolist()
    taguri = []
    for list in df.loc[:, 'tags']:
        taguri.append(ast.literal_eval(list))

    tags = tagCount(taguri)

    tagDf = pd.DataFrame()
    tagDf['tags'] = tags
    tagDf.drop_duplicates(inplace=True, ignore_index=True)

    for i, lista in enumerate(taguri):
        temp = []
        for tag in tags:
            temp.append(1 if tag in lista else -1)
        tagDf[str(ficuri[i])] = pd.Series(temp)
    print(tagDf.columns.__len__())
    print(ficuri.__len__())
    words = ['wordcount']
    maxW = max(df.loc[:,'words'])
    #print(maxW) #105146
    for i in ficuri:
        words= words + [df.loc[df['id']==i,'words'].values.tolist()[0]/maxW]
    tagDf = pd.DataFrame([words], columns=tagDf.columns).append(tagDf)
    tagDf.to_csv('tags_tiny.csv', index=False)

def recMatrix(kudos,tags): #kudos and tags in id x feature form
    kudosM = kudos.T.to_numpy()
    print(kudosM)
    tagsM = tags.to_numpy()
    print(tagsM)
    usersM = np.matmul(kudosM, tagsM)
    print(usersM)
    tagsM = tags.T.to_numpy()
    expected = np.matmul(usersM, tagsM)
    movie_rec = pd.DataFrame(data=expected, index=kudos.columns, columns=tags.T.columns)
    movie_rec.to_csv('recs.csv')

useri=pd.read_csv('useri.csv',header=0)
kudos = pd.read_csv('kudos_tiny.csv',header = 0)


tabel_taguri(kudos)
tags = pd.read_csv('tags_tiny.csv',header = 0)

kudos = kudos.T #id x users
tags = tags.T #id x tags


kudos.columns = kudos.iloc[0]
kudos.drop(kudos.index[0],inplace=True)
kudos.set_index = kudos.iloc[:,0]
kudos.sort_index(inplace=True)

tags.columns = tags.iloc[0]
tags.drop(tags.index[0],inplace=True)
tags.set_index = tags.iloc[:,0]
tags.sort_index(inplace=True)

kudos = kudos.loc[kudos.index.isin(tags.index.tolist()), :]
tags = tags.loc[tags.index.isin(kudos.index.tolist()), :]

kudosM = kudos.T.to_numpy() #users x id
tagsM = tags.to_numpy() #id x tags
usersM = np.matmul(kudosM, tagsM) #user x tag
usersM = usersM.transpose() #t x u
uN = len(usersM[0]) #u
tN = len(usersM)
iN=len(tagsM)
usersM[0] = [usersM[0][u]/sum(kudosM[u]) for u in range(uN)]
for t in range(1,tN):
    usersM[t] = [usersM[t][u]/sum(kudosM[u]) - sum(tagsM.T[t])/iN for u in range(uN)]
usersM = usersM.transpose() #u x t

userList = kudos.columns.tolist() #u
ficList = kudos.T.columns.tolist() #i
tagList = tags.columns.tolist() #t
userFicList = [userList, ficList]

ficCol = tagList #t
userCol = [tag+'_avg' for tag in tagList] #t

pairCols=['user','id','kudos']+ficCol #3 + t
pairIndex = pd.MultiIndex.from_product(userFicList, names=["user", "fic"]) # u1f1 u1f2
x=pairIndex.__len__() #498292
with open('pairs.csv', 'a', newline="") as f_out:
    writer = csv.writer(f_out)
    with open('errors_pairs.csv', 'a', newline="") as e_out:
        errorwriter = csv.writer(e_out)
        count=-1
        writer.writerow(pairCols)
        for i in pairIndex:
            k = 1 if kudos.loc[i[1], i[0]] == 1 else 0
            row = [x * y for x, y in zip(usersM[userList.index(i[0])].tolist(),tagsM[ficList.index(i[1])].tolist())]
            row = [i[0],i[1],k]+ row
            count=count+1
            print(count * 100 // x)
            try:
                writer.writerow(row)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                error_row = [id] + [sys.exc_info()[0]]
                errorwriter.writerow(error_row)
